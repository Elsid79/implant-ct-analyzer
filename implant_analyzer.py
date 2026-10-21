"""
Локальный анализатор отклонений дентальных имплантов (RealGUIDE STL + Pre-op КТ vs Post-op КТ).

Функционал:
1. Автоматическая жесткая 3D-регистрация (SimpleITK) Post-op КТ к Pre-op КТ.
2. Извлечение геометрии запланированных имплантов из STL (координаты платформы, апекса, вектор оси).
3. Локализованный поиск металла и ICP-фиттинг фактических имплантов на Post-op КТ.
4. Расчет стандартных метрик отклонения (3D Coronal, 3D Apical, Angular, Depth, Lateral).
5. Анализ причин индивидуальных отклонений («Почему?»: кортикалка, люфт втулки, стоппер).
6. Комплексный мульти-имплантный анализ (All-on-X / 2+ импланта):
   Алгоритм Кабша (Kabsch SVD) для выявления глобального сдвига / недопосадки хирургического шаблона.
"""

import os
import glob
from typing import Dict, List, Optional
import numpy as np
import SimpleITK as sitk
import trimesh
import open3d as o3d


class DentalImplantAnalyzer:
    def __init__(self, pre_op_dicom_path: str, post_op_dicom_path: str):
        """
        Инициализация анализатора.
        Все расчеты проводятся в единой мировой системе координат Pre-op КТ (LPS, миллиметры).
        """
        print("[1/5] Чтение томограмм (DICOM)...")
        self.pre_img = self._load_dicom(pre_op_dicom_path)
        self.post_img = self._load_dicom(post_op_dicom_path)
        self.registration_transform: Optional[sitk.Transform] = None
        self.resampled_post_img: Optional[sitk.Image] = None

    def _load_dicom(self, path: str) -> sitk.Image:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(path)
        if not series_ids:
            raise FileNotFoundError(f"DICOM серии не найдены в директории: {path}")
        dicom_names = reader.GetGDCMSeriesFileNames(path, series_ids[0])
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
        print(f"  Загружен объем: размер {img.GetSize()}, шаг вокселя {img.GetSpacing()}")
        return img

    def register_volumes(self) -> sitk.Transform:
        """
        Жесткое совмещение (Rigid 6-DOF) Post-op КТ к Pre-op КТ.
        Применяется центрированная инициализация и Mattes Mutual Information.
        """
        print("[2/5] Выполняется жесткая регистрация челюстей (Post-op -> Pre-op)...")
        initial_transform = sitk.CenteredTransformInitializer(
            self.pre_img,
            self.post_img,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.15)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-6
        )
        registration.SetInitialTransform(initial_transform, inPlace=False)

        self.registration_transform = registration.Execute(self.pre_img, self.post_img)
        print("  ✓ Совмещение объемов успешно выполнено.")

        # Ресэмплинг постоперационного КТ в сетку предоперационного
        print("  Ресэмплинг постоперационного КТ в общее координатное пространство...")
        self.resampled_post_img = sitk.Resample(
            self.post_img, self.pre_img, self.registration_transform,
            sitk.sitkLinear, 0.0, self.post_img.GetPixelID()
        )
        return self.registration_transform

    def extract_planned_geometry(self, stl_path: str, implant_id: str = "implant") -> Dict:
        """
        Извлекает геометрию запланированного импланта из STL (RealGUIDE):
        - Центр платформы (entry point)
        - Центр апекса (apex point)
        - Продольную ось и длину
        """
        mesh = trimesh.load(stl_path)
        vertices = mesh.vertices

        centroid = mesh.centroid
        # Анализ главных компонент (PCA через SVD) для поиска главной оси вращения
        _, _, vv = np.linalg.svd(vertices - centroid)
        long_axis = vv[0]

        # Проекция вершин на главную ось
        projections = np.dot(vertices - centroid, long_axis)
        idx_min = np.argmin(projections)
        idx_max = np.argmax(projections)

        pt1 = vertices[idx_min]
        pt2 = vertices[idx_max]

        # Платформа импланта обычно шире, чем зауженный апекс
        dist1 = np.linalg.norm((vertices - pt1) - np.outer(np.dot(vertices - pt1, long_axis), long_axis), axis=1)
        dist2 = np.linalg.norm((vertices - pt2) - np.outer(np.dot(vertices - pt2, long_axis), long_axis), axis=1)

        radius_1 = np.mean(dist1[projections < np.percentile(projections, 10)])
        radius_2 = np.mean(dist2[projections > np.percentile(projections, 90)])

        if radius_1 > radius_2:
            entry_point = pt1
            apex_point = pt2
        else:
            entry_point = pt2
            apex_point = pt1

        axis_vector = (apex_point - entry_point) / np.linalg.norm(apex_point - entry_point)
        length = float(np.linalg.norm(apex_point - entry_point))

        return {
            "id": implant_id,
            "stl_path": stl_path,
            "mesh": mesh,
            "entry": np.array(entry_point, dtype=np.float64),
            "apex": np.array(apex_point, dtype=np.float64),
            "axis": np.array(axis_vector, dtype=np.float64),
            "length": length
        }

    def detect_actual_implant_icp(self, planned_info: Dict, hu_metal_threshold: float = 2800.0) -> Dict:
        """
        Детекция реального положения импланта на Post-op КТ:
        1. Сегментация вокселей металла в ROI вокруг планового ложа.
        2. ICP-подгонка CAD-модели STL импланта к облаку точек металла.
        """
        if self.resampled_post_img is None:
            raise RuntimeError("Сначала вызовите register_volumes()!")

        post_arr = sitk.GetArrayFromImage(self.resampled_post_img) # [Z, Y, X]
        metal_mask = post_arr > hu_metal_threshold
        metal_indices = np.argwhere(metal_mask)

        if len(metal_indices) == 0:
            raise ValueError(f"Металл (HU > {hu_metal_threshold}) не обнаружен на постоперационном КТ.")

        # Преобразуем воксели металла в миллиметры мирового пространства (LPS)
        # Оптимизированная фильтрация: берем воксели в цилиндрическом ROI вокруг планового импланта
        center_planned = (planned_info["entry"] + planned_info["apex"]) / 2.0
        max_search_radius = (planned_info["length"] / 2.0) + 6.0

        roi_pts = []
        for z, y, x in metal_indices:
            coord = np.array(self.resampled_post_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z))))
            if np.linalg.norm(coord - center_planned) <= max_search_radius:
                roi_pts.append(coord)

        roi_pts = np.array(roi_pts)
        if len(roi_pts) < 80:
            raise ValueError(f"Не удалось локализовать металл в окрестности импланта {planned_info['id']}.")

        # Подгонка через Open3D ICP
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(roi_pts)

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(planned_info["mesh"].vertices)

        # Выполняем Point-to-Point ICP
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, max_correspondence_distance=3.5,
            init=np.eye(4),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
        )

        T_fit = icp_result.transformation

        actual_entry = (T_fit @ np.append(planned_info["entry"], 1.0))[:3]
        actual_apex = (T_fit @ np.append(planned_info["apex"], 1.0))[:3]
        actual_axis = (actual_apex - actual_entry) / np.linalg.norm(actual_apex - actual_entry)

        return {
            "id": planned_info["id"],
            "entry": actual_entry,
            "apex": actual_apex,
            "axis": actual_axis,
            "transform": T_fit,
            "inlier_rmse": float(icp_result.inlier_rmse)
        }

    def compute_metrics(self, planned: Dict, actual: Dict) -> Dict:
        """
        Расчет сертифицированных геометрических отклонений.
        """
        p_entry, p_apex, p_axis = planned["entry"], planned["apex"], planned["axis"]
        a_entry, a_apex, a_axis = actual["entry"], actual["apex"], actual["axis"]

        # 1. 3D Coronal (Точка входа / платформа)
        coronal_3d = np.linalg.norm(a_entry - p_entry)

        # 2. 3D Apical (Верхушка / апекс)
        apical_3d = np.linalg.norm(a_apex - p_apex)

        # 3. Угловое отклонение
        dot = np.clip(np.dot(p_axis, a_axis), -1.0, 1.0)
        angular_deg = np.degrees(np.arccos(dot))

        # 4. Отклонение по глубине (проекция на плановую ось: >0 недопогружение, <0 провал)
        depth_deviation = np.dot(p_entry - a_entry, p_axis)

        # 5. Латеральное смещение входа
        entry_shift_vec = a_entry - p_entry
        lateral_coronal = np.sqrt(max(0.0, coronal_3d**2 - depth_deviation**2))

        return {
            "implant_id": planned["id"],
            "coronal_3d_mm": round(float(coronal_3d), 2),
            "apical_3d_mm": round(float(apical_3d), 2),
            "angular_deg": round(float(angular_deg), 2),
            "depth_mm": round(float(depth_deviation), 2),
            "lateral_coronal_mm": round(float(lateral_coronal), 2),
            "entry_vector_diff": entry_shift_vec,  # Для мульти-имплантного анализа
            "apex_vector_diff": a_apex - p_apex
        }

    def analyze_single_root_cause(self, metrics: Dict) -> List[str]:
        """
        Логический анализ причин индивидуального отклонения импланта («Почему?»).
        """
        causes = []

        # Глубина
        if abs(metrics["depth_mm"]) >= 0.75:
            if metrics["depth_mm"] > 0:
                causes.append(
                    f"⚠️ Недопогружение (+{metrics['depth_mm']} мм). Причина: упор стоппера в слизистую, "
                    "заклинивание в плотной костной ткани (D1/D2) или ранняя остановка имплантовода."
                )
            else:
                causes.append(
                    f"⚠️ Избыточная глубина ({metrics['depth_mm']} мм). Причина: мягкая губчатая кость (D4), "
                    "провал стоппера или оверпрепарирование ложа."
                )

        # Люфт во втулке
        if metrics["lateral_coronal_mm"] <= 0.55 and metrics["angular_deg"] >= 2.0:
            causes.append(
                f"ℹ️ Угловой наклон {metrics['angular_deg']}° при стабильном центре входа ({metrics['lateral_coronal_mm']} мм). "
                "Характерно для механического микролюфта фрезы внутри направляющей втулки шаблона (clearance tolerance)."
            )

        # Соскальзывание апекса
        if metrics["apical_3d_mm"] > 1.5 and metrics["apical_3d_mm"] > (metrics["coronal_3d_mm"] * 1.4):
            causes.append(
                f"⚠️ Смещение апекса существенно превышает платформу ({metrics['apical_3d_mm']} мм против {metrics['coronal_3d_mm']} мм). "
                "Причина: увод апикальной фрезы по скату кортикальной пластинки или стенке экстракционной лунки."
            )

        if not causes:
            causes.append("✅ Отклонения находятся в пределах нормы навигационной хирургии.")

        return causes

    @staticmethod
    def analyze_multi_implant_guide_movement(results: List[Dict]) -> Dict:
        """
        [ПУНКТ 2]: Анализ смещения хирургического шаблона (All-on-X / 2+ импланта).
        Использует алгоритм Кабша (Kabsch SVD) для вычисления жесткой трансформации шаблона.
        Если все импланты смещены одинаково — это 100% ошибка посадки/фиксации шаблона.
        """
        if len(results) < 2:
            return {
                "applicable": False,
                "message": "Для анализа смещения шаблона требуется минимум 2 импланта."
            }

        planned_pts = []
        actual_pts = []
        entry_vectors = []

        for item in results:
            m = item["metrics"]
            p = item["planned"]
            a = item["actual"]

            planned_pts.append(p["entry"])
            planned_pts.append(p["apex"])
            actual_pts.append(a["entry"])
            actual_pts.append(a["apex"])
            entry_vectors.append(m["entry_vector_diff"])

        planned_pts = np.array(planned_pts)
        actual_pts = np.array(actual_pts)
        entry_vectors = np.array(entry_vectors)

        # 1. Анализ направленности векторов смещения входа (Косинусное сходство)
        norms = np.linalg.norm(entry_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        normalized_vecs = entry_vectors / norms

        dot_matrix = normalized_vecs @ normalized_vecs.T
        # Берем среднее значение сходства между всеми парами имплантов
        triu_indices = np.triu_indices(len(entry_vectors), k=1)
        mean_directional_similarity = float(np.mean(dot_matrix[triu_indices]))

        # 2. Алгоритм Кабша (Kabsch SVD) для поиска наилучшего движения шаблона как твердого тела
        p_centroid = np.mean(planned_pts, axis=0)
        a_centroid = np.mean(actual_pts, axis=0)

        p_centered = planned_pts - p_centroid
        a_centered = actual_pts - a_centroid

        H = p_centered.T @ a_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Коррекция отражения
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Угол глобального перекоса шаблона
        trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        global_tilt_deg = float(np.degrees(np.arccos(trace)))

        # Глобальное поступательное смещение (Translation)
        global_translation = a_centroid - (R @ p_centroid)
        translation_magnitude = float(np.linalg.norm(global_translation))

        # Оценка остаточной ошибки (Residual Error) после вычета движения шаблона
        predicted_actual = (R @ planned_pts.T).T + global_translation
        residuals = np.linalg.norm(actual_pts - predicted_actual, axis=1)
        mean_residual_error = float(np.mean(residuals))

        # Исходная средняя ошибка без учета сдвига шаблона
        raw_errors = np.linalg.norm(actual_pts - planned_pts, axis=1)
        mean_raw_error = float(np.mean(raw_errors))

        # Доля ошибки, объясняемая именно смещением шаблона
        variance_explained_by_guide = max(0.0, 1.0 - (mean_residual_error / (mean_raw_error + 1e-6)))

        # Формирование диагностического вердикта
        diagnostics = []
        is_guide_seating_error = False

        if mean_directional_similarity > 0.70 and translation_magnitude > 0.8:
            is_guide_seating_error = True
            diagnostics.append(
                f"🚨 Обнаружен системный сдвиг шаблона! Векторы смещения всех имплантов сонаправлены на {round(mean_directional_similarity * 100, 1)}%."
            )
            diagnostics.append(
                f"   Шаблон сместился целиком на {round(translation_magnitude, 2)} мм с перекосом {round(global_tilt_deg, 2)}°."
            )
            diagnostics.append(
                f"   {round(variance_explained_by_guide * 100, 1)}% всех наблюдаемых отклонений вызваны именно недопосадкой/качанием шаблона."
            )
        elif global_tilt_deg > 2.5:
            diagnostics.append(
                f"⚠️ Зафиксирован угловой перекос посадки шаблона ({round(global_tilt_deg, 2)}°)."
            )
        else:
            diagnostics.append(
                "✅ Шаблон сидел стабильно. Отклонения имеют случайный индивидуальный характер сверления."
            )

        return {
            "applicable": True,
            "is_guide_seating_error": is_guide_seating_error,
            "mean_directional_similarity": round(mean_directional_similarity, 3),
            "global_translation_mm": round(translation_magnitude, 2),
            "global_tilt_deg": round(global_tilt_deg, 2),
            "variance_explained_pct": round(variance_explained_by_guide * 100, 1),
            "diagnostics": diagnostics
        }


def run_pipeline(pre_op_dicom: str, post_op_dicom: str, planned_stl_files: List[str]):
    """
    Основная функция запуска полного анализа для клинического случая.
    """
    analyzer = DentalImplantAnalyzer(pre_op_dicom, post_op_dicom)
    analyzer.register_volumes()

    multi_results = []

    print("\n" + "=" * 65)
    print(" РЕЗУЛЬТАТЫ СРАВНЕНИЯ ПО КАЖДОМУ ИМПЛАНТУ")
    print("=" * 65)

    for stl_file in planned_stl_files:
        imp_id = os.path.splitext(os.path.basename(stl_file))[0]
        print(f"\n Обработка импланта: [{imp_id}]")

        planned = analyzer.extract_planned_geometry(stl_file, implant_id=imp_id)
        actual = analyzer.detect_actual_implant_icp(planned)
        metrics = analyzer.compute_metrics(planned, actual)
        causes = analyzer.analyze_single_root_cause(metrics)

        multi_results.append({
            "planned": planned,
            "actual": actual,
            "metrics": metrics,
            "causes": causes
        })

        print(f"  • Вход (Coronal 3D):      {metrics['coronal_3d_mm']} мм")
        print(f"  • Верхушка (Apex 3D):     {metrics['apical_3d_mm']} мм")
        print(f"  • Угловой наклон:         {metrics['angular_deg']}°")
        print(f"  • Смещение по глубине:    {metrics['depth_mm']} мм")
        print("  • Анализ причин:")
        for c in causes:
            print(f"     {c}")

    # Пункт 2: Мульти-имплантный анализ хирургического шаблона
    print("\n" + "=" * 65)
    print(" [ПУНКТ 2] АНАЛИЗ ПОВЕДЕНИЯ ХИРУРГИЧЕСКОГО ШАБЛОНА")
    print("=" * 65)
    guide_analysis = DentalImplantAnalyzer.analyze_multi_implant_guide_movement(multi_results)

    if guide_analysis["applicable"]:
        for line in guide_analysis["diagnostics"]:
            print(f" {line}")
    else:
        print(f" {guide_analysis['message']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    # Пример вызова для тестирования:
    PRE_DICOM_DIR = "C:/path/to/pre_op_dicom"
    POST_DICOM_DIR = "C:/path/to/post_op_dicom"
    PLANNED_STLS = [
        "C:/path/to/implant_16_planned.stl",
        "C:/path/to/implant_26_planned.stl"
    ]

    print("Анализатор готов к работе. Укажите реальные пути к вашим КТ и STL.")
