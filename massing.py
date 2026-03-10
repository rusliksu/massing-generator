#!/usr/bin/env python3
"""
Massing Generator — AI-генератор массинга зданий.
DWG на вход → DWG на выход.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import ezdxf
import yaml
from anthropic import Anthropic
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def load_config(config_path: str) -> dict:
    """Загрузка конфигурации из YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def convert_dwg_to_dxf(dwg_path: str, output_dir: str, oda_path: str = "ODAFileConverter") -> str:
    """Конвертация DWG → DXF через ODA File Converter."""
    input_dir = str(Path(dwg_path).parent)
    filename = Path(dwg_path).stem

    subprocess.run([
        oda_path,
        input_dir,
        output_dir,
        "ACAD2018", "DXF", "0", "1",
        f"*.dwg"
    ], check=True)

    dxf_path = Path(output_dir) / f"{filename}.dxf"
    if not dxf_path.exists():
        raise FileNotFoundError(f"Конвертация не удалась: {dxf_path}")
    return str(dxf_path)


def convert_dxf_to_dwg(dxf_path: str, output_dir: str, oda_path: str = "ODAFileConverter") -> str:
    """Конвертация DXF → DWG через ODA File Converter."""
    input_dir = str(Path(dxf_path).parent)
    filename = Path(dxf_path).stem

    subprocess.run([
        oda_path,
        input_dir,
        output_dir,
        "ACAD2018", "DWG", "0", "1",
        "*.dxf"
    ], check=True)

    dwg_path = Path(output_dir) / f"{filename}.dwg"
    if not dwg_path.exists():
        raise FileNotFoundError(f"Конвертация не удалась: {dwg_path}")
    return str(dwg_path)


def parse_site_boundary(dxf_path: str, layer: str = None, scale: float = 1.0) -> dict:
    """
    Извлечение границ пятна застройки из DXF.
    Возвращает полигон с координатами и метаданные.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    polygons = []

    # Ищем замкнутые полилинии (LWPOLYLINE, POLYLINE)
    for entity in msp:
        if layer and entity.dxf.layer != layer:
            continue

        if entity.dxftype() == "LWPOLYLINE" and entity.closed:
            points = [(p[0], p[1]) for p in entity.get_points()]
            if len(points) >= 3:
                polygons.append(Polygon(points))

        elif entity.dxftype() == "POLYLINE" and entity.is_closed:
            points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
            if len(points) >= 3:
                polygons.append(Polygon(points))

    if not polygons:
        # Попробуем найти по всем слоям, если конкретный не задан
        if layer:
            print(f"Слой '{layer}' не найден или пуст. Поиск по всем слоям...")
            return parse_site_boundary(dxf_path, layer=None)
        raise ValueError("Не найдены замкнутые полилинии в DXF")

    # Берём самый большой полигон как пятно застройки
    site = max(polygons, key=lambda p: p.area)

    # Масштабируем координаты (например, мм → м при scale=0.001)
    from shapely.affinity import scale as shapely_scale
    site_scaled = shapely_scale(site, xfact=scale, yfact=scale, origin=(0, 0))

    bounds = site_scaled.bounds  # (minx, miny, maxx, maxy)
    coords = list(site_scaled.exterior.coords)

    return {
        "coordinates": coords,
        "coordinates_original": list(site.exterior.coords),  # для записи DXF
        "scale": scale,
        "area_m2": round(site_scaled.area, 2),
        "bounds": {
            "min_x": round(bounds[0], 2),
            "min_y": round(bounds[1], 2),
            "max_x": round(bounds[2], 2),
            "max_y": round(bounds[3], 2),
        },
        "width": round(bounds[2] - bounds[0], 2),
        "height": round(bounds[3] - bounds[1], 2),
        "centroid": {
            "x": round(site_scaled.centroid.x, 2),
            "y": round(site_scaled.centroid.y, 2),
        },
        "all_polygons_count": len(polygons),
    }


def parse_existing_buildings(dxf_path: str, scale: float = 1.0) -> list[dict]:
    """
    Извлекает существующие здания из DXF.
    Ищет замкнутые полилинии на слоях с маркерами зданий/этажей.
    """
    from shapely.affinity import scale as shapely_scale

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Слои, содержащие существующие здания
    building_layer_patterns = ["к1", "к2", "этаж", "инт13", "инт14"]
    # Кадастровые слои: маленькие участки (<500 м²) — вероятно здания
    cadastral_layer_patterns = ["p_"]

    buildings = []
    for entity in msp:
        if entity.dxftype() != "LWPOLYLINE" or not entity.closed:
            continue

        layer = entity.dxf.layer.lower()
        points = [(p[0], p[1]) for p in entity.get_points()]
        if len(points) < 3:
            continue

        poly = Polygon(points)
        if poly.area <= 0:
            continue

        poly_scaled = shapely_scale(poly, xfact=scale, yfact=scale, origin=(0, 0))
        area = poly_scaled.area

        # Прямое совпадение со слоями зданий
        is_building = any(p in layer for p in building_layer_patterns)

        # Маленькие кадастровые участки (<500 м²) — скорее всего существующие постройки
        is_small_parcel = (any(layer.startswith(p) for p in cadastral_layer_patterns)
                          and 50 < area < 500)

        if (is_building and 50 < area < 50000) or is_small_parcel:
            buildings.append({
                "layer": entity.dxf.layer,
                "polygon": poly_scaled,
                "area_m2": round(area, 1),
                "centroid": (round(poly_scaled.centroid.x, 1), round(poly_scaled.centroid.y, 1)),
            })

    return buildings


def parse_cadastral_parcels(dxf_path: str, scale: float = 1.0) -> list[dict]:
    """Извлекает кадастровые участки из DXF (слои p_*)."""
    from shapely.affinity import scale as shapely_scale

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    parcels = []
    for entity in msp:
        if entity.dxftype() != "LWPOLYLINE" or not entity.closed:
            continue
        if not entity.dxf.layer.lower().startswith("p_"):
            continue

        points = [(p[0], p[1]) for p in entity.get_points()]
        if len(points) < 3:
            continue

        poly = Polygon(points)
        if poly.area <= 0:
            continue

        poly_scaled = shapely_scale(poly, xfact=scale, yfact=scale, origin=(0, 0))
        area = poly_scaled.area

        if area > 500:  # только участки > 500 м² (не постройки)
            parcels.append({
                "layer": entity.dxf.layer,
                "polygon": poly_scaled,
                "area_m2": round(area, 1),
            })

    return parcels


def parse_roads(dxf_path: str, scale: float = 1.0) -> list[Polygon]:
    """
    Извлекает дороги из DXF (слой ГП Дорога и похожие).
    Возвращает список полигонов дорог.
    """
    from shapely.affinity import scale as shapely_scale

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    road_patterns = ["дорог", "road", "проезд", "улиц"]
    roads = []

    for entity in msp:
        layer = entity.dxf.layer.lower()
        is_road = any(p in layer for p in road_patterns)
        if not is_road:
            continue

        if entity.dxftype() == "LWPOLYLINE":
            points = [(p[0], p[1]) for p in entity.get_points()]
            if len(points) < 2:
                continue

            from shapely.geometry import LineString
            line = LineString(points)
            line_scaled = shapely_scale(line, xfact=scale, yfact=scale, origin=(0, 0))

            # Буферизуем линию в полигон дороги (ширина ~6м с каждой стороны)
            road_poly = line_scaled.buffer(6)
            if road_poly.area > 10:
                roads.append(road_poly)

        elif entity.dxftype() == "LWPOLYLINE" and entity.closed:
            points = [(p[0], p[1]) for p in entity.get_points()]
            if len(points) >= 3:
                poly = Polygon(points)
                poly_scaled = shapely_scale(poly, xfact=scale, yfact=scale, origin=(0, 0))
                if poly_scaled.area > 10:
                    roads.append(poly_scaled)

    return roads


def compute_buildable_area(site_data: dict, config: dict,
                           existing_buildings: list = None,
                           roads: list = None) -> dict:
    """Вычисляет зону застройки с учётом отступов, существующих зданий и дорог."""
    site_polygon = Polygon(site_data["coordinates"])
    setback = config.get("setbacks", {}).get("default", 6)
    road_setback = config.get("setbacks", {}).get("road", 10)
    buildable = site_polygon.buffer(-setback)

    # Берём из site_data если не переданы явно
    if existing_buildings is None:
        existing_buildings = site_data.get("existing_buildings", [])
    if roads is None:
        roads = site_data.get("roads", [])

    # Вырезаем существующие здания (с буфером 15м — пожарный разрыв)
    if existing_buildings:
        for eb in existing_buildings:
            exclusion = eb["polygon"].buffer(15)
            buildable = buildable.difference(exclusion)
            if buildable.is_empty:
                break

    # Вырезаем дороги (с увеличенным отступом)
    if roads:
        for road in roads:
            exclusion = road.buffer(road_setback)
            buildable = buildable.difference(exclusion)
            if buildable.is_empty:
                break

    # Если после вырезания осталось MultiPolygon — берём самый большой кусок
    if buildable.geom_type == 'MultiPolygon':
        buildable = max(buildable.geoms, key=lambda g: g.area)

    # Упрощаем для промпта (меньше вершин)
    simplified = buildable.simplify(2.0, preserve_topology=True)
    if simplified.geom_type == 'MultiPolygon':
        simplified = max(simplified.geoms, key=lambda g: g.area)
    coords = list(simplified.exterior.coords)
    bounds = simplified.bounds

    return {
        "polygon": buildable,
        "simplified_coords": [(round(c[0], 1), round(c[1], 1)) for c in coords],
        "area_m2": round(buildable.area, 2),
        "bounds": bounds,
        "width": round(bounds[2] - bounds[0], 2),
        "height": round(bounds[3] - bounds[1], 2),
    }


def generate_parcel_based_layout(site_data: dict, config: dict) -> list[dict]:
    """
    Генерирует раскладку зданий по кадастровым участкам.
    Каждый участок получает одно здание, вписанное внутрь с отступами.
    Форма зависит от размера/пропорций участка.
    """
    import numpy as np
    from shapely.geometry import box
    from shapely.affinity import rotate
    import random

    parcels = site_data.get("parcels", [])
    if not parcels:
        print("  Нет кадастровых участков — фоллбэк на кварталы")
        return []

    # Определяем угол участка для ориентации зданий
    site_poly = Polygon(site_data["coordinates"])
    mrr = site_poly.minimum_rotated_rectangle
    mrr_coords = list(mrr.exterior.coords)
    edge1 = np.array(mrr_coords[1]) - np.array(mrr_coords[0])
    edge2 = np.array(mrr_coords[2]) - np.array(mrr_coords[1])
    if np.linalg.norm(edge1) > np.linalg.norm(edge2):
        main_angle = np.degrees(np.arctan2(edge1[1], edge1[0]))
    else:
        main_angle = np.degrees(np.arctan2(edge2[1], edge2[0]))

    buildable_info = compute_buildable_area(site_data, config)
    buildable = buildable_info["polygon"]

    buildings = []
    building_id = 1

    for parcel in parcels:
        pp = parcel["polygon"]
        area = parcel["area_m2"]

        # Пропускаем слишком маленькие участки
        if area < 500:
            continue

        # Участок должен быть внутри buildable зоны хотя бы на 50%
        if buildable.intersection(pp).area / pp.area < 0.5:
            continue

        # Для больших участков (>8000 м²) — делим на sub-зоны и ставим несколько зданий
        if area > 8000:
            from shapely.ops import split
            from shapely.geometry import LineString

            # Разбиваем пополам по длинной оси
            p_mrr = pp.minimum_rotated_rectangle
            pc = list(p_mrr.exterior.coords)
            pe1_len = np.linalg.norm(np.array(pc[1]) - np.array(pc[0]))
            pe2_len = np.linalg.norm(np.array(pc[2]) - np.array(pc[1]))
            if pe1_len > pe2_len:
                mid1 = ((pc[0][0] + pc[3][0]) / 2, (pc[0][1] + pc[3][1]) / 2)
                mid2 = ((pc[1][0] + pc[2][0]) / 2, (pc[1][1] + pc[2][1]) / 2)
            else:
                mid1 = ((pc[0][0] + pc[1][0]) / 2, (pc[0][1] + pc[1][1]) / 2)
                mid2 = ((pc[2][0] + pc[3][0]) / 2, (pc[2][1] + pc[3][1]) / 2)

            # Добавляем sub-parcels обратно в список для обработки
            try:
                cut_line = LineString([mid1, mid2]).buffer(0.5)
                sub = pp.difference(cut_line)
                if sub.geom_type == 'MultiPolygon':
                    for sp in sub.geoms:
                        if sp.area > 500:
                            parcels.append({"polygon": sp, "area_m2": round(sp.area, 1), "layer": parcel.get("layer", "split")})
                    continue
            except Exception:
                pass  # если не получилось разбить — ставим одно здание

        cx, cy = pp.centroid.x, pp.centroid.y

        # Определяем пропорции участка через MRR
        p_mrr = pp.minimum_rotated_rectangle
        p_coords = list(p_mrr.exterior.coords)
        pe1 = np.linalg.norm(np.array(p_coords[1]) - np.array(p_coords[0]))
        pe2 = np.linalg.norm(np.array(p_coords[2]) - np.array(p_coords[1]))
        p_long = max(pe1, pe2)
        p_short = min(pe1, pe2)

        # Отступ от границ участка (6м)
        setback = 6
        avail_long = p_long - setback * 2
        avail_short = p_short - setback * 2

        if avail_long < 20 or avail_short < 10:
            continue  # слишком маленький для здания

        # Глубина корпуса 12-18м (жилой дом)
        depth = min(18, avail_short * 0.75)
        depth = max(12, depth)
        if depth > avail_short:
            depth = avail_short

        # Выбираем форму здания по размеру участка
        if area > 4000 and avail_long > 60:
            # Большой участок — Н-образное (два крыла + перемычка)
            shape_type = random.choices(["h_shape", "rect", "l_shape"], weights=[50, 30, 20], k=1)[0]
        elif area > 2000:
            shape_type = random.choices(["rect", "l_shape", "h_shape"], weights=[50, 30, 20], k=1)[0]
        else:
            shape_type = random.choices(["rect", "l_shape"], weights=[70, 30], k=1)[0]

        # Длина здания — заполняем ~70-85% доступной длины
        b_len = avail_long * random.uniform(0.70, 0.85)
        b_len = max(30, min(b_len, 120))  # 30-120м

        hw, hd = b_len / 2, depth / 2

        if shape_type == "h_shape":
            # Н-образное: два крыла + перемычка посередине
            wing_len = b_len * 0.38  # каждое крыло ~38% общей длины
            gap = b_len - wing_len * 2  # перемычка
            conn_d = depth * 0.4  # глубина перемычки

            fp = Polygon([
                # Левое крыло
                (cx - hw, cy - hd),
                (cx - hw + wing_len, cy - hd),
                (cx - hw + wing_len, cy - conn_d / 2),
                # Перемычка нижняя
                (cx + hw - wing_len, cy - conn_d / 2),
                # Правое крыло
                (cx + hw - wing_len, cy - hd),
                (cx + hw, cy - hd),
                (cx + hw, cy + hd),
                (cx + hw - wing_len, cy + hd),
                (cx + hw - wing_len, cy + conn_d / 2),
                # Перемычка верхняя
                (cx - hw + wing_len, cy + conn_d / 2),
                (cx - hw + wing_len, cy + hd),
                (cx - hw, cy + hd),
            ])

        elif shape_type == "l_shape":
            # Г-образное
            wing_short = b_len * 0.45
            fp = Polygon([
                (cx - hw, cy - hd),
                (cx - hw + wing_short, cy - hd),
                (cx - hw + wing_short, cy - hd + depth * 0.4),
                (cx + hw, cy - hd + depth * 0.4),
                (cx + hw, cy + hd),
                (cx - hw, cy + hd),
            ])

        else:
            # Прямоугольник (вытянутый 1:4 - 1:6)
            fp = box(cx - hw, cy - hd, cx + hw, cy + hd)

        # Поворот по главной оси участка
        fp_rotated = rotate(fp, main_angle, origin=(cx, cy))

        # Проверка что здание внутри участка (с допуском)
        overlap = pp.intersection(fp_rotated).area / fp_rotated.area
        if overlap < 0.7:
            # Пробуем уменьшить
            fp_small = rotate(box(cx - hw * 0.7, cy - hd, cx + hw * 0.7, cy + hd),
                              main_angle, origin=(cx, cy))
            if pp.intersection(fp_small).area / fp_small.area >= 0.7:
                fp_rotated = fp_small
                shape_type = "rect_small"
            else:
                continue

        # Проверка на пересечение с уже размещёнными зданиями (мин. 8м — мягкий)
        skip = False
        for existing in buildings:
            ep = Polygon(existing["footprint"])
            if fp_rotated.distance(ep) < 8:
                skip = True
                break
        if skip:
            continue

        coords = list(fp_rotated.exterior.coords)[:-1]
        buildings.append({
            "id": building_id,
            "footprint": [[round(c[0], 1), round(c[1], 1)] for c in coords],
            "area_m2": round(fp_rotated.area, 0),
            "shape": shape_type,
            "parcel_area": round(area, 0),
            "orientation_deg": round(main_angle, 1),
            "cx": round(cx, 1),
            "cy": round(cy, 1),
        })
        building_id += 1

    print(f"  Размещено {len(buildings)} зданий в кадастровых участках")
    shapes = {}
    for b in buildings:
        shapes[b["shape"]] = shapes.get(b["shape"], 0) + 1
    if shapes:
        print(f"  Формы: {', '.join(f'{k}={v}' for k, v in shapes.items())}")

    return buildings


def _make_l_shape(cx, cy, wing_w, wing_h, depth):
    """Создаёт Г-образный полигон."""
    return Polygon([
        (cx - wing_w/2, cy - wing_h/2),
        (cx + wing_w/2, cy - wing_h/2),
        (cx + wing_w/2, cy - wing_h/2 + depth),
        (cx - wing_w/2 + depth, cy - wing_h/2 + depth),
        (cx - wing_w/2 + depth, cy + wing_h/2),
        (cx - wing_w/2, cy + wing_h/2),
    ])


def _make_u_shape(cx, cy, total_w, total_h, depth, gap):
    """Создаёт П-образный полигон (двор внутри)."""
    hw, hh = total_w/2, total_h/2
    return Polygon([
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx + hw, cy + hh),
        (cx + hw - depth, cy + hh),
        (cx + hw - depth, cy - hh + depth),
        (cx - hw + depth, cy - hh + depth),
        (cx - hw + depth, cy + hh),
        (cx - hw, cy + hh),
    ])


def generate_courtyard_blocks(site_data: dict, config: dict) -> list[dict]:
    """
    Генерирует квартальные блоки (двор + 3-5 секций вокруг).
    Варьирует размеры: 70-110м блок, разная глубина крыльев.
    Реалистичная квартальная застройка с Г-образными и П-образными вариациями.
    """
    import numpy as np
    from shapely.geometry import box
    from shapely.affinity import rotate
    import random

    # seed задаётся снаружи (main) для поддержки --variants

    buildable_info = compute_buildable_area(site_data, config)
    buildable = buildable_info["polygon"]
    bounds = buildable.bounds

    # Определяем угол участка
    mrr = buildable.minimum_rotated_rectangle
    mrr_coords = list(mrr.exterior.coords)
    edge1 = np.array(mrr_coords[1]) - np.array(mrr_coords[0])
    edge2 = np.array(mrr_coords[2]) - np.array(mrr_coords[1])
    if np.linalg.norm(edge1) > np.linalg.norm(edge2):
        main_angle = np.degrees(np.arctan2(edge1[1], edge1[0]))
    else:
        main_angle = np.degrees(np.arctan2(edge2[1], edge2[0]))

    # Шаблоны кварталов разного размера (компактнее для плотной застройки)
    block_templates = [
        {"w": 85, "h": 70, "depth": 15, "name": "wide"},
        {"w": 70, "h": 85, "depth": 15, "name": "tall"},
        {"w": 75, "h": 75, "depth": 16, "name": "square"},
        {"w": 90, "h": 65, "depth": 14, "name": "long"},
    ]

    max_block = max(max(t["w"], t["h"]) for t in block_templates)
    # Учитываем вращение: повёрнутые здания выступают за номинальные границы
    rotation_margin = max_block * abs(np.sin(np.radians(main_angle))) * 0.5
    grid_step = max_block + 18 + rotation_margin  # квартал + проезд + поправка на вращение

    x_range = np.arange(bounds[0] + 50, bounds[2] - 50, grid_step)
    y_range = np.arange(bounds[1] + 50, bounds[3] - 50, grid_step)

    blocks = []
    block_id = 1

    for ix, cx in enumerate(x_range):
        for iy, cy in enumerate(y_range):
            # Чередуем шаблоны для разнообразия
            tmpl = block_templates[(ix + iy) % len(block_templates)]
            hw, hh = tmpl["w"] / 2, tmpl["h"] / 2
            d = tmpl["depth"]

            buildings_in_block = []

            # Выбираем схему квартала (classic чаще — надёжнее проходит валидацию)
            scheme = random.choices(
                ["classic", "l_corners", "u_wrap"],
                weights=[60, 25, 15],
                k=1
            )[0]

            if scheme == "l_corners":
                # Два Г-образных здания (углы) + 1-2 прямые секции
                # Уменьшенные крылья чтобы влезать в зону
                wing = hh * 0.55  # длина крыла ~55% от полувысоты блока
                # Верхний левый угол (Г-образный)
                tl = Polygon([
                    (cx - hw, cy + hh),
                    (cx - hw + d + wing, cy + hh),
                    (cx - hw + d + wing, cy + hh - d),
                    (cx - hw + d, cy + hh - d),
                    (cx - hw + d, cy + hh - wing),
                    (cx - hw, cy + hh - wing),
                ])
                # Нижний правый угол (Г-образный)
                br = Polygon([
                    (cx + hw, cy - hh),
                    (cx + hw - d - wing, cy - hh),
                    (cx + hw - d - wing, cy - hh + d),
                    (cx + hw - d, cy - hh + d),
                    (cx + hw - d, cy - hh + wing),
                    (cx + hw, cy - hh + wing),
                ])
                # Прямые секции для заполнения оставшихся сторон
                top_fill = box(cx - hw + d + wing, cy + hh - d, cx + hw, cy + hh)
                bot_fill = box(cx - hw, cy - hh, cx + hw - d - wing, cy - hh + d)
                sections = [("l_corner_tl", tl), ("l_corner_br", br),
                           ("top_fill", top_fill)]
                if random.random() > 0.4:
                    sections.append(("bot_fill", bot_fill))

            elif scheme == "u_wrap":
                # П-образное: три стороны квартала одним зданием
                u_shape = Polygon([
                    (cx - hw, cy - hh),
                    (cx + hw, cy - hh),
                    (cx + hw, cy + hh - d),
                    (cx + hw - d, cy + hh - d),
                    (cx + hw - d, cy - hh + d),
                    (cx - hw + d, cy - hh + d),
                    (cx - hw + d, cy + hh - d),
                    (cx - hw, cy + hh - d),
                ])
                # Отдельная секция сверху (закрывает двор)
                closer = box(cx - hw, cy + hh - d, cx + hw, cy + hh)
                sections = [("u_wrap", u_shape)]
                if random.random() > 0.4:
                    sections.append(("closer", closer))

            else:
                # Classic — прямоугольные секции (как раньше)
                top = box(cx - hw, cy + hh - d, cx + hw, cy + hh)
                bot_shrink = random.choice([0, 0.15, 0.25])
                bottom = box(cx - hw * (1 - bot_shrink), cy - hh,
                             cx + hw * (1 - bot_shrink), cy - hh + d)
                left = box(cx - hw, cy - hh + d, cx - hw + d, cy + hh - d)
                right = box(cx + hw - d, cy - hh + d, cx + hw, cy + hh - d)
                sections = [("top", top), ("bottom", bottom), ("left", left)]
                if random.random() > 0.3:
                    sections.append(("right", right))

            for label, shape in sections:
                rotated = rotate(shape, main_angle, origin=(cx, cy))
                # Более мягкая проверка: 90% площади внутри (краевые здания)
                if buildable.contains(rotated) or (
                    buildable.intersection(rotated).area / rotated.area >= 0.85
                ):
                    # Клиппим к buildable если не полностью внутри
                    if not buildable.contains(rotated):
                        clipped = buildable.intersection(rotated)
                        if clipped.geom_type == 'Polygon' and clipped.area > rotated.area * 0.5:
                            rotated = clipped
                        else:
                            continue
                    fp = list(rotated.exterior.coords)[:-1]
                    buildings_in_block.append({
                        "position": label,
                        "footprint": [[round(p[0], 1), round(p[1], 1)] for p in fp],
                        "area_m2": round(rotated.area, 0),
                        "orientation_deg": round(main_angle, 1),
                    })

            # Квартал валиден если хотя бы 2 здания внутри
            if len(buildings_in_block) >= 2:
                blocks.append({
                    "block_id": block_id,
                    "center": [round(cx, 1), round(cy, 1)],
                    "buildings": buildings_in_block,
                    "building_count": len(buildings_in_block),
                    "template": tmpl["name"],
                })
                block_id += 1

    print(f"  Сгенерировано {len(blocks)} кварталов ({sum(b['building_count'] for b in blocks)} зданий)")
    return blocks


def generate_internal_roads(blocks: list[dict], site_data: dict) -> list[dict]:
    """
    Генерирует внутренние проезды между квартальными блоками.
    Соединяет центры соседних блоков полосами шириной 7м.
    """
    from shapely.geometry import LineString
    import numpy as np

    if len(blocks) < 2:
        return []

    site_poly = Polygon(site_data["coordinates"])
    centers = [(b["center"][0], b["center"][1]) for b in blocks]

    roads = []
    connected = set()

    # Соединяем каждый блок с ближайшими 2 соседями
    for i, c1 in enumerate(centers):
        distances = []
        for j, c2 in enumerate(centers):
            if i == j:
                continue
            d = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            distances.append((j, d))
        distances.sort(key=lambda x: x[1])

        for j, dist in distances[:2]:
            pair = tuple(sorted([i, j]))
            if pair in connected:
                continue
            connected.add(pair)

            c2 = centers[j]
            line = LineString([c1, c2])
            road_poly = line.buffer(3.5)  # 7м ширина

            # Обрезаем по границе участка
            road_clipped = road_poly.intersection(site_poly)
            if road_clipped.is_empty or road_clipped.area < 10:
                continue

            roads.append({
                "from_block": blocks[i]["block_id"],
                "to_block": blocks[j]["block_id"],
                "polygon": road_clipped,
                "length_m": round(dist, 1),
            })

    return roads


def generate_building_slots(site_data: dict, config: dict) -> list[dict]:
    """
    Генерирует сетку допустимых позиций для зданий внутри buildable polygon.
    Включает прямоугольные, Г-образные и П-образные здания.
    """
    import numpy as np
    from shapely.geometry import box
    from shapely.affinity import rotate

    buildable_info = compute_buildable_area(site_data, config)
    buildable = buildable_info["polygon"]
    fire_dist = config.get("fire_safety", {}).get("min_distance", 12)

    bounds = buildable.bounds

    # Определяем основное направление участка
    mrr = buildable.minimum_rotated_rectangle
    mrr_coords = list(mrr.exterior.coords)
    edge1 = np.array(mrr_coords[1]) - np.array(mrr_coords[0])
    edge2 = np.array(mrr_coords[2]) - np.array(mrr_coords[1])
    if np.linalg.norm(edge1) > np.linalg.norm(edge2):
        main_angle = np.degrees(np.arctan2(edge1[1], edge1[0]))
    else:
        main_angle = np.degrees(np.arctan2(edge2[1], edge2[0]))

    slots = []
    slot_id = 1

    # Плотная сетка: шаг = ширина здания + пожарный разрыв
    grid_step = 18 + fire_dist  # 30 м — по глубине секции

    x_range = np.arange(bounds[0] + 40, bounds[2] - 40, grid_step)
    y_range = np.arange(bounds[1] + 40, bounds[3] - 40, grid_step)

    # Типы зданий
    building_templates = [
        # (тип, описание, генератор полигона)
        ("rect", "80x16 секционный", lambda cx, cy: box(cx-40, cy-8, cx+40, cy+8)),
        ("rect", "60x16 секционный", lambda cx, cy: box(cx-30, cy-8, cx+30, cy+8)),
        ("rect", "45x16 секционный", lambda cx, cy: box(cx-22.5, cy-8, cx+22.5, cy+8)),
        ("rect", "20x20 башня", lambda cx, cy: box(cx-10, cy-10, cx+10, cy+10)),
        ("L", "Г-образный 45x45", lambda cx, cy: _make_l_shape(cx, cy, 45, 45, 16)),
        ("U", "П-образный 60x45", lambda cx, cy: _make_u_shape(cx, cy, 60, 45, 16, 28)),
    ]

    for bt_type, bt_desc, bt_gen in building_templates:
        for cx in x_range:
            for cy in y_range:
                shape = bt_gen(cx, cy)
                rotated = rotate(shape, main_angle, origin='centroid')

                if buildable.contains(rotated):
                    fp = list(rotated.exterior.coords)[:-1]
                    slots.append({
                        "slot_id": slot_id,
                        "center": [round(cx, 1), round(cy, 1)],
                        "type": bt_type,
                        "size": bt_desc,
                        "footprint": [[round(p[0], 1), round(p[1], 1)] for p in fp],
                        "area_m2": round(rotated.area, 0),
                        "orientation_deg": round(main_angle, 1),
                    })
                    slot_id += 1

    print(f"  Сгенерировано {len(slots)} допустимых слотов для зданий")
    return slots


def generate_infill_buildings(site_data: dict, config: dict, blocks: list[dict]) -> list[dict]:
    """Генерирует отдельные здания в пустых зонах между кварталами."""
    import numpy as np
    from shapely.geometry import box, Point
    from shapely.affinity import rotate

    buildable_info = compute_buildable_area(site_data, config)
    buildable = buildable_info["polygon"]
    fire_dist = config.get("fire_safety", {}).get("min_distance", 12)

    # Собираем все полигоны зданий кварталов
    block_polys = []
    for bl in blocks:
        for b in bl["buildings"]:
            block_polys.append(Polygon(b["footprint"]))

    # Определяем угол
    mrr = buildable.minimum_rotated_rectangle
    mrr_coords = list(mrr.exterior.coords)
    edge1 = np.array(mrr_coords[1]) - np.array(mrr_coords[0])
    edge2 = np.array(mrr_coords[2]) - np.array(mrr_coords[1])
    if np.linalg.norm(edge1) > np.linalg.norm(edge2):
        main_angle = np.degrees(np.arctan2(edge1[1], edge1[0]))
    else:
        main_angle = np.degrees(np.arctan2(edge2[1], edge2[0]))

    bounds = buildable.bounds
    infill = []
    infill_id = 1000  # чтобы не конфликтовать с block buildings

    # Шаблоны infill-зданий (меньше чем квартальные)
    templates = [
        ("60x16 секционный", lambda cx, cy: box(cx-30, cy-8, cx+30, cy+8)),
        ("45x16 секционный", lambda cx, cy: box(cx-22.5, cy-8, cx+22.5, cy+8)),
        ("24x24 башня", lambda cx, cy: box(cx-12, cy-12, cx+12, cy+12)),
    ]

    grid_step = 40  # плотная сетка для infill
    x_range = np.arange(bounds[0] + 30, bounds[2] - 30, grid_step)
    y_range = np.arange(bounds[1] + 30, bounds[3] - 30, grid_step)

    for cx in x_range:
        for cy in y_range:
            # Проверяем что точка далеко от всех блочных зданий
            pt = Point(cx, cy)
            too_close = False
            for bp in block_polys:
                if pt.distance(bp) < fire_dist + 30:  # 30м буфер от кварталов
                    too_close = True
                    break
            if too_close:
                continue

            # Пробуем разные шаблоны
            for desc, gen in templates:
                shape = gen(cx, cy)
                rotated = rotate(shape, main_angle, origin='centroid')
                if buildable.contains(rotated):
                    # Проверяем fire distance до всех существующих infill
                    ok = True
                    for existing in infill:
                        ep = Polygon(existing["footprint"])
                        if rotated.distance(ep) < fire_dist:
                            ok = False
                            break
                    if ok:
                        fp = list(rotated.exterior.coords)[:-1]
                        infill.append({
                            "infill_id": infill_id,
                            "center": [round(cx, 1), round(cy, 1)],
                            "size": desc,
                            "footprint": [[round(p[0], 1), round(p[1], 1)] for p in fp],
                            "area_m2": round(rotated.area, 0),
                            "orientation_deg": round(main_angle, 1),
                        })
                        infill_id += 1
                        break  # один шаблон на позицию

    return infill


def build_prompt(site_data: dict, config: dict, slots: list[dict] = None,
                 blocks: list[dict] = None, infill: list[dict] = None,
                 parcel_buildings: list[dict] = None) -> str:
    """Формирование промпта для AI."""
    constraints = config.get("constraints", {})
    fire = config.get("fire_safety", {})
    insol = config.get("insolation", {})
    region = config.get("site", {}).get("region", "moscow")
    latitude = config.get("site", {}).get("latitude", 55.75)

    buildable = compute_buildable_area(site_data, config)

    # Режим парсельной раскладки (приоритет — как у Ильи)
    if parcel_buildings:
        buildings_text = ""
        for b in parcel_buildings:
            buildings_text += (f"  Здание {b['id']}: форма={b['shape']}, пятно={b['area_m2']}м², "
                              f"участок={b['parcel_area']}м², footprint={json.dumps(b['footprint'])}\n")

        total_footprint = sum(b["area_m2"] for b in parcel_buildings)
        return f"""Ты — архитектор-градостроитель. Назначь этажность готовым зданиям.

## Участок
- Площадь: {site_data['area_m2']} м² | Зона застройки: {buildable['area_m2']} м²
- Регион: {region} | Широта: {latitude}°

## Здания (уже размещены в кадастровых участках, {len(parcel_buildings)} шт):
{buildings_text}
- Суммарное пятно: {total_footprint:.0f} м²

## Ограничения
- Макс. плотность: {constraints.get('max_density', 2.5)}
- Этажность: 9-{constraints.get('max_floors', 25)} | Высота этажа: 3.0 м
- Sellable ≈ 78% от gross
- Линейная застройка: центр выше, края ниже

## Задача
Для КАЖДОГО из {len(parcel_buildings)} зданий назначь этажность (9-25).
- Здания с большим пятном (>1500м²): 14-25 этажей
- Средние (800-1500м²): 12-20 этажей
- Малые (<800м²): 9-15 этажей
- h_shape: 16-25 этажей (доминанты)
- В центре участка выше, по краям ниже
- Варьируй: соседние дома не одинаковой высоты

## Формат: ТОЛЬКО JSON
{{
  "buildings": [
    {{
      "id": 1,
      "type": "residential",
      "footprint": [[x,y],...],
      "floors": 17,
      "floor_height": 3.0,
      "total_height": 51.0,
      "gross_area": 18360,
      "sellable_area": 14321,
      "orientation_deg": {parcel_buildings[0]['orientation_deg'] if parcel_buildings else 0}
    }}
  ],
  "summary": {{
    "total_buildings": {len(parcel_buildings)},
    "total_gross_area": 150000,
    "total_sellable_area": 117000,
    "density": 1.5,
    "site_coverage_ratio": 0.20,
    "notes": "описание решения"
  }}
}}"""

    # Режим кварталов
    if blocks:
        blocks_text = ""
        for bl in blocks:
            blocks_text += f"\nКвартал {bl['block_id']} (центр {bl['center']}, {bl['building_count']} зданий):\n"
            for b in bl["buildings"]:
                blocks_text += f"  - {b['position']}: {b['area_m2']}м², footprint={json.dumps(b['footprint'])}\n"

        infill_text = ""
        if infill:
            infill_text = "\n\n## Отдельные здания (infill, в пустых зонах между кварталами):\n"
            for inf in infill:
                infill_text += f"  - infill_{inf['infill_id']}: {inf['size']}, {inf['area_m2']}м², footprint={json.dumps(inf['footprint'])}\n"

        return f"""Ты — архитектор-градостроитель. Назначь этажность зданиям в готовых кварталах и отдельным зданиям.

## Участок
- Площадь: {site_data['area_m2']} м² | Зона застройки: {buildable['area_m2']} м²

## Готовые кварталы (здания уже размещены, все внутри зоны):
{blocks_text}{infill_text}

## Ограничения
- Макс. плотность: {constraints.get('max_density', 2.5)} | Целевая площадь: {constraints.get('target_area', 45000)} м²
- Этажность: 9-{constraints.get('max_floors', 25)} | Высота этажа: {constraints.get('min_floor_height', 3.0)} м
- Sellable ≈ 75-80% от gross

## Задача
Для КАЖДОГО здания (квартальных и infill) назначь этажность.
- Квартальные верхние/нижние (длинные) секции: 14-25 этажей
- Квартальные боковые (короткие) секции: 9-17 этажей (ниже длинных, для инсоляции двора)
- Infill-здания: 12-20 этажей (секционные), 20-25 этажей (башни)
- Выше в центре участка, ниже по краям
- Варьируй этажность
- ОБЯЗАТЕЛЬНО сохрани block_id для квартальных зданий. Для infill — block_id: null
- ВКЛЮЧИ ВСЕ infill-здания в ответ

## Формат: ТОЛЬКО JSON
{{
  "buildings": [
    {{
      "id": 1,
      "block_id": 1,
      "position": "top",
      "type": "residential",
      "footprint": [[x,y],...],
      "floors": 17,
      "floor_height": 3.0,
      "total_height": 51.0,
      "gross_area": 24480,
      "sellable_area": 19584,
      "orientation_deg": 15
    }}
  ],
  "summary": {{
    "total_buildings": 25,
    "total_gross_area": 400000,
    "total_sellable_area": 320000,
    "density": 2.3,
    "site_coverage_ratio": 0.30,
    "notes": "описание"
  }}
}}"""

    # Если есть слоты — режим выбора из готовых позиций
    if slots:
        slots_summary = []
        for s in slots:
            slots_summary.append(f"  Слот {s['slot_id']}: центр={s['center']}, размер={s['size']}, площадь={s['area_m2']}м²")
        slots_text = "\n".join(slots_summary)

        return f"""Ты — архитектор-градостроитель. Выбери оптимальный набор зданий из предложенных позиций.

## Участок
- Площадь: {site_data['area_m2']} м²
- Площадь зоны застройки (с отступами): {buildable['area_m2']} м²

## Доступные позиции для зданий (все гарантированно внутри зоны застройки):
{slots_text}

## Ограничения
- Макс. плотность застройки: {constraints.get('max_density', 2.5)}
- Целевая продаваемая площадь: {constraints.get('target_area', 45000)} м²
- Макс. этажность: {constraints.get('max_floors', 25)}
- Высота этажа: {constraints.get('min_floor_height', 3.0)} м
- Sellable area ≈ 75-80% от gross area
- Мин. расстояние между зданиями: {fire.get('min_distance', 12)} м (уже учтено в сетке)
- Регион: {region}, широта: {latitude}° — учитывай инсоляцию

## Задача
Выбери 15-25 слотов и назначь этажность. Цель: плотная квартальная застройка, максимум продаваемой площади.

Правила:
- Формируй дворовые пространства: группируй 3-4 здания вокруг двора (секционные + Г/П-образные)
- Предпочитай длинные секционные дома (80x16, 60x16) и Г/П-образные — это реальная застройка
- Башни (20x20) — как акценты на углах или входах, не больше 2-3 штук
- Распределяй по ВСЕЙ территории равномерно
- Варьируй этажность: 9-{constraints.get('max_floors', 25)} этажей (выше в центре, ниже по краям)
- ВАЖНО: не выбирай слоты, которые пересекаются (проверяй footprint координаты!)
- Выбирай МНОГО зданий (15-25), не жадничай

## Формат ответа
Верни ТОЛЬКО валидный JSON (без markdown):
{{
  "buildings": [
    {{
      "id": 1,
      "slot_id": 5,
      "type": "residential",
      "footprint": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "floors": 17,
      "floor_height": 3.0,
      "total_height": 51.0,
      "gross_area": 18360,
      "sellable_area": 14688,
      "orientation_deg": 15
    }}
  ],
  "summary": {{
    "total_buildings": 8,
    "total_gross_area": 150000,
    "total_sellable_area": 120000,
    "density": 2.3,
    "site_coverage_ratio": 0.25,
    "notes": "описание решения"
  }}
}}"""

    # Фоллбэк — старый режим без слотов
    buildable_coords = buildable["simplified_coords"]

    return f"""Ты — архитектор-градостроитель. Сгенерируй оптимальный массинг для жилой застройки.

## Зона застройки (отступы уже учтены!)
- Координаты зоны (в метрах): {json.dumps(buildable_coords)}
- Площадь зоны: {buildable['area_m2']} м²
- Площадь участка (до отступов): {site_data['area_m2']} м²

ВСЕ footprint-координаты зданий ДОЛЖНЫ быть СТРОГО ВНУТРИ зоны застройки.

## Ограничения
- Макс. плотность: {constraints.get('max_density', 2.5)} | Целевая площадь: {constraints.get('target_area', 45000)} м²
- Макс. этажность: {constraints.get('max_floors', 25)} | Высота этажа: {constraints.get('min_floor_height', 3.0)} м
- Sellable ≈ 75-80% от gross | Мин. разрыв: {fire.get('min_distance', 12)} м
- Регион: {region}, широта: {latitude}°

## Формат: ТОЛЬКО валидный JSON
{{
  "buildings": [{{ "id": 1, "type": "residential", "footprint": [[x1,y1],...], "floors": 22,
    "floor_height": 3.0, "total_height": 66.0, "gross_area": 15000, "sellable_area": 12000, "orientation_deg": 15 }}],
  "summary": {{ "total_buildings": 3, "total_gross_area": 50000, "total_sellable_area": 45000,
    "density": 2.3, "site_coverage_ratio": 0.35, "notes": "описание" }}
}}"""


def generate_massing(prompt: str, api_key: str = None, base_url: str = None) -> dict:
    """Генерация массинга через Claude API."""
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = Anthropic(**client_kwargs)

    # Retry на connection/timeout errors (VPS прокси может дропнуть)
    import time as _time
    max_api_retries = 5
    for attempt in range(max_api_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                timeout=120.0,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as e:
            err_name = type(e).__name__
            is_retriable = any(k in err_name for k in ("Connection", "Timeout", "APIConnection", "APITimeout"))
            if is_retriable and attempt < max_api_retries - 1:
                delay = 10 * (attempt + 1)  # 10, 20, 30, 40с
                print(f"  Ошибка соединения, повтор через {delay}с... ({err_name})")
                _time.sleep(delay)
                continue
            raise

    text = response.content[0].text

    # Парсим JSON из ответа
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Попробуем извлечь JSON из markdown
        import re
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Не удалось распарсить JSON из ответа AI:\n{text}")


def compute_sun_position(latitude: float, hour: float, day_of_year: int = 81) -> tuple[float, float]:
    """
    Вычисляет азимут и высоту солнца.
    day_of_year=81 → 22 марта (равноденствие, нормативная дата для инсоляции).
    Возвращает (azimuth_deg, altitude_deg). Азимут от юга по часовой.
    """
    import math

    # Склонение солнца (формула Купера)
    declination = 23.45 * math.sin(math.radians(360 / 365 * (284 + day_of_year)))
    decl_rad = math.radians(declination)
    lat_rad = math.radians(latitude)

    # Часовой угол (15° на час от солнечного полудня ~12:30 для Москвы)
    hour_angle = (hour - 12.5) * 15
    ha_rad = math.radians(hour_angle)

    # Высота солнца
    sin_alt = (math.sin(lat_rad) * math.sin(decl_rad) +
               math.cos(lat_rad) * math.cos(decl_rad) * math.cos(ha_rad))
    altitude = math.degrees(math.asin(max(-1, min(1, sin_alt))))

    # Азимут (от юга, по часовой = положительный на запад)
    cos_az = ((math.sin(decl_rad) - math.sin(lat_rad) * sin_alt) /
              (math.cos(lat_rad) * math.cos(math.radians(altitude)) + 1e-10))
    azimuth = math.degrees(math.acos(max(-1, min(1, cos_az))))
    if hour_angle > 0:
        azimuth = -azimuth  # после полудня — на запад (отрицательный = восточнее юга)

    # Конвертируем в "от севера по часовой" (стандартный азимут)
    azimuth_from_north = 180 + azimuth

    return azimuth_from_north, altitude


def compute_shadow_polygon(footprint: list, height: float,
                           sun_azimuth: float, sun_altitude: float) -> Polygon:
    """
    Вычисляет полигон тени от здания.
    footprint: [[x,y], ...] — координаты основания
    height: высота здания в метрах
    sun_azimuth: азимут солнца (от севера, по часовой)
    sun_altitude: высота солнца в градусах
    """
    import math

    if sun_altitude <= 0:
        return Polygon()  # солнце за горизонтом

    # Длина тени
    shadow_length = height / math.tan(math.radians(sun_altitude))

    # Направление тени (противоположно солнцу)
    shadow_dir = math.radians(sun_azimuth + 180)
    dx = shadow_length * math.sin(shadow_dir)
    dy = shadow_length * math.cos(shadow_dir)

    # Полигон тени = объединение footprint и сдвинутого footprint
    base = Polygon(footprint)
    shifted_pts = [(p[0] + dx, p[1] + dy) for p in footprint]
    shifted = Polygon(shifted_pts)

    shadow = unary_union([base, shifted]).convex_hull
    return shadow


def check_insolation(massing: dict, config: dict) -> list[dict]:
    """
    Проверяет инсоляцию: для каждого здания считает часы прямого солнца.
    Возвращает список зданий с нарушениями (< min_hours).

    Метод: каждые 30 мин с 7:00 до 17:00 на 22 марта считаем тень от всех зданий.
    Если центроид фасада затенён — этот интервал не считается.
    """
    latitude = config.get("site", {}).get("latitude", 55.75)
    min_hours = config.get("insolation", {}).get("min_hours", 2.0)
    buildings = massing.get("buildings", [])

    if not buildings:
        return []

    # Точки проверки: 7:00-17:00, шаг 30 мин = 20 интервалов
    time_slots = [7.0 + i * 0.5 for i in range(21)]  # 7.0, 7.5, ... 17.0
    step_hours = 0.5

    # Предвычисляем позицию солнца для каждого слота
    sun_positions = []
    for t in time_slots:
        az, alt = compute_sun_position(latitude, t)
        if alt > 2:  # солнце выше 2° (практический порог)
            sun_positions.append((t, az, alt))

    # Для каждого здания строим полигон и высоту
    building_data = []
    for b in buildings:
        fp = b["footprint"]
        height = b.get("total_height", b.get("floors", 15) * b.get("floor_height", 3.0))
        poly = Polygon(fp)
        building_data.append({"building": b, "polygon": poly, "height": height,
                              "footprint": fp})

    # Для каждого здания считаем часы солнца
    violations = []
    for i, bd in enumerate(building_data):
        target = bd["polygon"]
        target_centroid = target.centroid
        sun_hours = 0
        max_continuous = 0
        current_continuous = 0

        for t, az, alt in sun_positions:
            shaded = False
            # Проверяем тень от каждого ДРУГОГО здания
            for j, other in enumerate(building_data):
                if i == j:
                    continue
                shadow = compute_shadow_polygon(other["footprint"], other["height"], az, alt)
                if shadow.is_valid and shadow.contains(target_centroid):
                    shaded = True
                    break

            if not shaded:
                sun_hours += step_hours
                current_continuous += step_hours
                max_continuous = max(max_continuous, current_continuous)
            else:
                current_continuous = 0

        bd["building"]["sun_hours"] = round(sun_hours, 1)
        bd["building"]["max_continuous_sun"] = round(max_continuous, 1)

        if max_continuous < min_hours:
            violations.append({
                "building_id": bd["building"]["id"],
                "floors": bd["building"].get("floors", "?"),
                "sun_hours_total": round(sun_hours, 1),
                "max_continuous": round(max_continuous, 1),
                "required": min_hours,
            })

    return violations


def fix_insolation_violations(massing: dict, violations: list[dict], config: dict):
    """Снижает этажность зданий-нарушителей до устранения затенения."""
    if not violations:
        return

    violation_ids = {v["building_id"] for v in violations}
    buildings = massing.get("buildings", [])

    # Для нарушителей — пробуем снизить соседей на 2-3 этажа
    # Упрощённый подход: снижаем самые высокие здания рядом с нарушителями
    for v in violations:
        target = next((b for b in buildings if b["id"] == v["building_id"]), None)
        if not target:
            continue
        target_poly = Polygon(target["footprint"])

        # Находим ближайшие высокие здания
        neighbors = []
        for b in buildings:
            if b["id"] == v["building_id"]:
                continue
            bp = Polygon(b["footprint"])
            dist = target_poly.distance(bp)
            if dist < 100:  # в радиусе 100м
                neighbors.append((b, dist))

        neighbors.sort(key=lambda x: x[0].get("floors", 0), reverse=True)

        # Снижаем этажность самого высокого соседа на 3
        for nb, dist in neighbors[:2]:
            old_floors = nb.get("floors", 15)
            new_floors = max(9, old_floors - 3)
            if new_floors < old_floors:
                nb["floors"] = new_floors
                nb["total_height"] = new_floors * nb.get("floor_height", 3.0)
                nb["gross_area"] = round(Polygon(nb["footprint"]).area * new_floors)
                nb["sellable_area"] = round(nb["gross_area"] * 0.78)
                print(f"    Здание {nb['id']}: {old_floors}→{new_floors} эт. "
                      f"(затеняет здание {v['building_id']})")


def _inject_block_ids(massing: dict, blocks: list[dict]):
    """Сопоставляет здания из AI-ответа с блоками по центру блока."""
    from shapely.geometry import Point

    # Для каждого блока — центр и радиус
    block_centers = []
    for bl in blocks:
        cx, cy = bl["center"]
        block_centers.append((bl["block_id"], Point(cx, cy)))

    matched = 0
    for building in massing.get("buildings", []):
        # Если AI уже вернул block_id — проверяем что он валидный
        existing = building.get("block_id")
        if existing and any(bc[0] == existing for bc in block_centers):
            matched += 1
            continue

        bp = Polygon(building["footprint"])
        centroid = bp.centroid
        best_dist = float("inf")
        best_block = None
        for block_id, center in block_centers:
            d = centroid.distance(center)
            if d < best_dist:
                best_dist = d
                best_block = block_id
        # Привязываем если центроид здания в пределах 70м от центра блока
        if best_block is not None and best_dist < 70:
            building["block_id"] = best_block
            matched += 1

    print(f"  Block ID injection: {matched}/{len(massing.get('buildings', []))} зданий привязаны к кварталам")


def clip_massing_to_buildable(massing: dict, site_data: dict, config: dict) -> dict:
    """Удаляет здания с overlap <50% и пересекающиеся здания."""
    buildable_info = compute_buildable_area(site_data, config)
    buildable_area = buildable_info["polygon"]
    fire_dist = config.get("fire_safety", {}).get("min_distance", 12)

    # Шаг 1: убрать здания вне зоны
    valid_buildings = []
    removed = 0
    for b in massing.get("buildings", []):
        footprint = Polygon(b["footprint"])
        if footprint.area <= 0:
            continue
        overlap = buildable_area.intersection(footprint).area / footprint.area
        if overlap >= 0.50:
            valid_buildings.append(b)
        else:
            removed += 1
            print(f"    Удалено здание {b['id']} (overlap {overlap*100:.0f}%)")

    if removed:
        print(f"    Удалено вне зоны: {removed}")

    # Шаг 2: убрать пересекающиеся (оставляем здание с большей площадью)
    # Здания внутри одного квартала (block_id) освобождены от fire distance
    kept = []
    for b in valid_buildings:
        fp = Polygon(b["footprint"])
        conflict = False
        for k in kept:
            # Пропускаем проверку для зданий одного квартала
            b_block = b.get("block_id")
            k_block = k.get("block_id")
            if b_block is not None and k_block is not None and b_block == k_block:
                continue

            kfp = Polygon(k["footprint"])
            if fp.distance(kfp) < fire_dist:
                # Конфликт — оставляем большее
                if b.get("sellable_area", 0) <= k.get("sellable_area", 0):
                    conflict = True
                    print(f"    Удалено здание {b['id']} (конфликт с {k['id']}, dist={fp.distance(kfp):.1f}м)")
                    break
                else:
                    kept.remove(k)
                    print(f"    Удалено здание {k['id']} (конфликт с {b['id']}, меньше площадь)")
        if not conflict:
            kept.append(b)

    if len(kept) < len(valid_buildings):
        print(f"    Удалено пересечений: {len(valid_buildings) - len(kept)}")

    valid_buildings = kept

    massing["buildings"] = valid_buildings
    # Обновляем summary
    total_gross = sum(b.get("gross_area", 0) for b in valid_buildings)
    total_sell = sum(b.get("sellable_area", 0) for b in valid_buildings)
    site_area = site_data["area_m2"]
    massing["summary"] = {
        "total_buildings": len(valid_buildings),
        "total_gross_area": total_gross,
        "total_sellable_area": total_sell,
        "density": round(total_gross / site_area, 2) if site_area > 0 else 0,
        "site_coverage_ratio": massing.get("summary", {}).get("site_coverage_ratio", 0),
        "notes": massing.get("summary", {}).get("notes", ""),
    }
    return massing


def validate_massing(massing: dict, site_data: dict, config: dict, overlap_threshold: float = 0.75) -> list[str]:
    """Валидация сгенерированного массинга."""
    errors = []
    constraints = config.get("constraints", {})
    fire = config.get("fire_safety", {})

    buildable_info = compute_buildable_area(site_data, config)
    buildable_area = buildable_info["polygon"]

    buildings_polygons = []

    for b in massing.get("buildings", []):
        footprint = Polygon(b["footprint"])
        buildings_polygons.append(footprint)

        # Проверка: здание внутри пятна с tolerance
        if not buildable_area.contains(footprint):
            overlap = buildable_area.intersection(footprint).area / footprint.area if footprint.area > 0 else 0
            if overlap < overlap_threshold:
                errors.append(
                    f"Здание {b['id']}: {overlap*100:.0f}% внутри зоны застройки (нужно ≥{overlap_threshold*100:.0f}%)"
                )

        # Проверка этажности
        max_floors = constraints.get("max_floors", 25)
        if b.get("floors", 0) > max_floors:
            errors.append(f"Здание {b['id']}: этажность {b['floors']} > макс. {max_floors}")

    # Проверка противопожарных разрывов (внутри одного квартала — пропуск)
    min_dist = fire.get("min_distance", 12)
    buildings_list = massing.get("buildings", [])
    for i in range(len(buildings_polygons)):
        for j in range(i + 1, len(buildings_polygons)):
            # Здания одного квартала — fire distance не проверяем
            bi_block = buildings_list[i].get("block_id") if i < len(buildings_list) else None
            bj_block = buildings_list[j].get("block_id") if j < len(buildings_list) else None
            if bi_block is not None and bj_block is not None and bi_block == bj_block:
                continue

            dist = buildings_polygons[i].distance(buildings_polygons[j])
            if dist < min_dist:
                errors.append(
                    f"Здания {i+1} и {j+1}: расстояние {dist:.1f}м < мин. {min_dist}м "
                    f"(blocks: {bi_block} vs {bj_block})"
                )

    # Проверка плотности
    summary = massing.get("summary", {})
    max_density = constraints.get("max_density", 2.5)
    if summary.get("density", 0) > max_density:
        errors.append(f"Плотность {summary['density']} > макс. {max_density}")

    return errors


def write_massing_to_dxf(massing: dict, site_data: dict, output_path: str):
    """Запись массинга в DXF файл."""
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()

    scale = site_data.get("scale", 1.0)
    inv_scale = 1.0 / scale if scale != 0 else 1.0  # м → оригинальные единицы

    # Создаём слои
    doc.layers.add("SITE_BOUNDARY", color=7)       # Белый — граница участка
    doc.layers.add("MASSING_BUILDINGS", color=1)    # Красный — здания
    doc.layers.add("MASSING_INFO", color=3)         # Зелёный — информация
    doc.layers.add("EXISTING_BUILDINGS", color=8)   # Серый — существующие здания
    doc.layers.add("ROADS", color=42)               # Жёлтый — дороги
    doc.layers.add("CADASTRAL", color=94)           # Зелёный — кадастр
    doc.layers.add("INTERNAL_ROADS", color=252)     # Светло-серый — проезды
    doc.layers.add("MASSING_SHADOWS", color=170)    # Синий — тени

    # Рисуем границу участка в оригинальных единицах
    site_coords = site_data.get("coordinates_original", site_data["coordinates"])
    msp.add_lwpolyline(
        [(c[0], c[1]) for c in site_coords],
        dxfattribs={"layer": "SITE_BOUNDARY"},
        close=True,
    )

    # Существующие здания
    for eb in site_data.get("existing_buildings", []):
        coords = list(eb["polygon"].exterior.coords)
        msp.add_lwpolyline(
            [(c[0] * inv_scale, c[1] * inv_scale) for c in coords],
            dxfattribs={"layer": "EXISTING_BUILDINGS"},
            close=True,
        )

    # Дороги
    for road in site_data.get("roads", []):
        if road.geom_type == 'Polygon':
            coords = list(road.exterior.coords)
            msp.add_lwpolyline(
                [(c[0] * inv_scale, c[1] * inv_scale) for c in coords],
                dxfattribs={"layer": "ROADS"},
                close=True,
            )

    # Кадастровые участки
    for parcel in site_data.get("parcels", []):
        coords = list(parcel["polygon"].exterior.coords)
        msp.add_lwpolyline(
            [(c[0] * inv_scale, c[1] * inv_scale) for c in coords],
            dxfattribs={"layer": "CADASTRAL"},
            close=True,
        )

    # Внутренние проезды
    for iroad in site_data.get("internal_roads", []):
        road_poly = iroad["polygon"]
        polys = [road_poly] if road_poly.geom_type == 'Polygon' else list(road_poly.geoms)
        for rp in polys:
            coords = list(rp.exterior.coords)
            msp.add_lwpolyline(
                [(c[0] * inv_scale, c[1] * inv_scale) for c in coords],
                dxfattribs={"layer": "INTERNAL_ROADS"},
                close=True,
            )

    # Рисуем здания (AI генерирует в метрах, конвертируем обратно)
    for b in massing.get("buildings", []):
        footprint = b["footprint"]

        # Контур здания (м → оригинальные единицы)
        msp.add_lwpolyline(
            [(p[0] * inv_scale, p[1] * inv_scale) for p in footprint],
            dxfattribs={"layer": "MASSING_BUILDINGS"},
            close=True,
        )

        # Подпись: этажность и площадь
        centroid = Polygon(footprint).centroid
        label = f"{b.get('floors', '?')}эт. / {b.get('sellable_area', '?')}м²"
        # Размер текста в оригинальных единицах
        text_height_orig = max(site_data["width"], site_data["height"]) * inv_scale / 50
        msp.add_text(
            label,
            dxfattribs={
                "layer": "MASSING_INFO",
                "height": text_height_orig,
                "insert": (centroid.x * inv_scale, centroid.y * inv_scale),
            },
        )

    # Сводная информация
    summary = massing.get("summary", {})
    # Позиция в оригинальных единицах
    orig_bounds = site_data.get("coordinates_original", site_data["coordinates"])
    orig_poly = Polygon(orig_bounds)
    ob = orig_poly.bounds
    info_y = ob[1] - (ob[3] - ob[1]) * 0.15
    info_x = ob[0]
    text_height = (max(ob[2] - ob[0], ob[3] - ob[1])) / 40

    info_lines = [
        f"Всего зданий: {summary.get('total_buildings', '?')}",
        f"Продаваемая площадь: {summary.get('total_sellable_area', '?')} м²",
        f"Плотность: {summary.get('density', '?')}",
        f"Покрытие: {summary.get('site_coverage_ratio', '?')}",
        f"Квартиры: ~{summary.get('est_apartments', '?')} (ср. 55 м²)",
        f"Паркинг: ~{summary.get('est_parking_spots', '?')} м/м",
        f"Жители: ~{summary.get('est_residents', '?')} чел.",
    ]

    for i, line in enumerate(info_lines):
        msp.add_text(
            line,
            dxfattribs={
                "layer": "MASSING_INFO",
                "height": text_height,
                "insert": (info_x, info_y - i * text_height * 1.5),
            },
        )

    doc.saveas(output_path)
    print(f"DXF сохранён: {output_path}")


def visualize_massing(massing: dict, site_data: dict, output_path: str = None, config: dict = None):
    """Визуализация массинга через matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    # Граница участка
    site_poly = Polygon(site_data["coordinates"])
    x, y = site_poly.exterior.xy
    ax.plot(x, y, 'b-', linewidth=2, label='Граница участка')

    # Buildable area (с вырезами под существующие здания и дороги)
    from shapely.geometry import Polygon as ShapelyPolygon
    buildable_info = compute_buildable_area(site_data, {"setbacks": {"default": 6, "road": 10}})
    buildable = buildable_info["polygon"]
    if buildable.geom_type == 'Polygon':
        bx, by = buildable.exterior.xy
        ax.plot(bx, by, 'g--', linewidth=1, alpha=0.5, label='Зона застройки')
    elif buildable.geom_type == 'MultiPolygon':
        for geom in buildable.geoms:
            bx, by = geom.exterior.xy
            ax.plot(bx, by, 'g--', linewidth=1, alpha=0.5)

    # Существующие здания (серые)
    for eb in site_data.get("existing_buildings", []):
        ex, ey = eb["polygon"].exterior.xy
        ax.fill(ex, ey, facecolor='gray', edgecolor='darkgray', linewidth=1, alpha=0.5)
    if site_data.get("existing_buildings"):
        ax.plot([], [], color='gray', linewidth=5, alpha=0.5, label=f'Существующие ({len(site_data["existing_buildings"])})')

    # Кадастровые участки (зелёный пунктир, тонкий)
    for parcel in site_data.get("parcels", []):
        px, py = parcel["polygon"].exterior.xy
        ax.plot(px, py, color='green', linewidth=0.6, linestyle=':', alpha=0.4)
    if site_data.get("parcels"):
        ax.plot([], [], color='green', linewidth=1, linestyle=':', alpha=0.5,
                label=f'Кадастр ({len(site_data["parcels"])})')

    # Дороги (светло-коричневые)
    for road in site_data.get("roads", []):
        if road.geom_type == 'Polygon':
            rx, ry = road.exterior.xy
            ax.fill(rx, ry, facecolor='wheat', edgecolor='tan', linewidth=0.5, alpha=0.4)
    if site_data.get("roads"):
        ax.plot([], [], color='wheat', linewidth=5, alpha=0.5, label=f'Дороги ({len(site_data["roads"])})')

    # Внутренние проезды (светло-серые)
    for iroad in site_data.get("internal_roads", []):
        road_poly = iroad["polygon"]
        if road_poly.geom_type == 'Polygon':
            irx, iry = road_poly.exterior.xy
            ax.fill(irx, iry, facecolor='lightgray', edgecolor='gray', linewidth=0.5, alpha=0.5)
        elif road_poly.geom_type == 'MultiPolygon':
            for g in road_poly.geoms:
                irx, iry = g.exterior.xy
                ax.fill(irx, iry, facecolor='lightgray', edgecolor='gray', linewidth=0.5, alpha=0.5)
    if site_data.get("internal_roads"):
        ax.plot([], [], color='lightgray', linewidth=5, alpha=0.6,
                label=f'Проезды ({len(site_data["internal_roads"])})')

    # Тени от зданий (полдень, 22 марта)
    latitude = config.get("site", {}).get("latitude", 54.6) if config else 54.6
    az_noon, alt_noon = compute_sun_position(latitude, 12.0)
    if alt_noon > 2:
        for b in massing.get("buildings", []):
            height = b.get("total_height", b.get("floors", 15) * b.get("floor_height", 3.0))
            shadow = compute_shadow_polygon(b["footprint"], height, az_noon, alt_noon)
            if shadow.is_valid and shadow.area > 0:
                if shadow.geom_type == 'Polygon':
                    sx, sy = shadow.exterior.xy
                    ax.fill(sx, sy, facecolor='navy', alpha=0.08)
        ax.plot([], [], color='navy', linewidth=5, alpha=0.15, label='Тени (12:00, 22 марта)')

    # Здания
    colors_by_block = {}
    cmap = plt.cm.Set3
    building_patches = []
    for b in massing.get("buildings", []):
        fp = b["footprint"]
        poly = plt.Polygon(fp, closed=True)
        block_id = b.get("block_id", 0)
        if block_id not in colors_by_block:
            colors_by_block[block_id] = cmap(len(colors_by_block) % 12)
        color = colors_by_block[block_id]

        ax.add_patch(plt.Polygon(fp, closed=True, facecolor=color, edgecolor='black',
                                  linewidth=1.5, alpha=0.7))

        # Подпись этажности + инсоляция
        centroid = Polygon(fp).centroid
        floors = b.get("floors", "?")
        sun_h = b.get("max_continuous_sun")
        label = f"{floors}эт"
        if sun_h is not None:
            label += f"\n{sun_h}ч"
            if sun_h < 2.0:
                # Красная обводка для нарушений инсоляции
                ax.add_patch(plt.Polygon(fp, closed=True, facecolor='none',
                                          edgecolor='red', linewidth=3, alpha=0.8))
        ax.text(centroid.x, centroid.y, label, ha='center', va='center',
                fontsize=6, fontweight='bold')

    ax.set_aspect('equal')

    summary = massing.get("summary", {})
    ax.set_title(f"Массинг — {summary.get('total_buildings', '?')} зданий, "
                 f"{summary.get('total_sellable_area', '?')} м²",
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Сводная таблица внизу
    info_parts = [
        f"Плотность: {summary.get('density', '?')}",
        f"Покрытие: {summary.get('site_coverage_ratio', '?')}",
    ]
    if summary.get("est_apartments"):
        info_parts.append(f"~{summary['est_apartments']} кв.")
        info_parts.append(f"~{summary.get('est_parking_spots', '?')} м/м")
        info_parts.append(f"~{summary.get('est_residents', '?')} жит.")
    fig.text(0.5, 0.01, " | ".join(info_parts), ha='center', fontsize=9, color='gray')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="AI Massing Generator")
    parser.add_argument("--input", "-i", required=True, help="DWG/DXF файл с пятном застройки")
    parser.add_argument("--config", "-c", default="config.yaml", help="Конфигурация (YAML)")
    parser.add_argument("--output", "-o", help="Выходной файл (по умолчанию: <input>_massing.dwg)")
    parser.add_argument("--density", "-d", type=float, help="Макс. плотность застройки")
    parser.add_argument("--target-area", "-a", type=float, help="Целевая продаваемая площадь, м²")
    parser.add_argument("--region", "-r", help="Регион")
    parser.add_argument("--api-key", help="Anthropic API key (или env ANTHROPIC_API_KEY)")
    parser.add_argument("--base-url", help="API base URL")
    parser.add_argument("--oda-path", default="ODAFileConverter", help="Путь к ODA File Converter")
    parser.add_argument("--max-retries", type=int, default=3, help="Макс. попыток при ошибках валидации")
    parser.add_argument("--layer", "-l", help="Слой DXF с пятном застройки")
    parser.add_argument("--scale", "-s", type=float, default=1.0,
                        help="Масштаб координат (0.001 для мм→м)")
    parser.add_argument("--variants", "-v", type=int, default=1,
                        help="Количество вариантов (1-5). Каждый с разным seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Файл не найден: {input_path}")
        sys.exit(1)

    # Загрузка конфига
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {"site": {}, "constraints": {}, "setbacks": {}, "fire_safety": {}, "insolation": {}}

    # CLI-параметры перезаписывают конфиг
    if args.density:
        config.setdefault("constraints", {})["max_density"] = args.density
    if args.target_area:
        config.setdefault("constraints", {})["target_area"] = args.target_area
    if args.region:
        config.setdefault("site", {})["region"] = args.region

    # Определяем формат и конвертируем если нужно
    with tempfile.TemporaryDirectory() as tmpdir:
        if input_path.suffix.lower() == ".dwg":
            print("Конвертация DWG → DXF...")
            dxf_path = convert_dwg_to_dxf(str(input_path), tmpdir, args.oda_path)
        elif input_path.suffix.lower() == ".dxf":
            dxf_path = str(input_path)
        else:
            print(f"Неподдерживаемый формат: {input_path.suffix}")
            sys.exit(1)

        # Парсинг пятна застройки
        print("Парсинг пятна застройки...")
        site_data = parse_site_boundary(dxf_path, layer=args.layer, scale=args.scale)
        print(f"  Площадь участка: {site_data['area_m2']} м²")
        print(f"  Размеры: {site_data['width']} x {site_data['height']} м")
        print(f"  Найдено полигонов: {site_data['all_polygons_count']}")

        # Парсинг существующих зданий, дорог, кадастровых участков
        print("\nПарсинг контекста участка...")
        existing_buildings = parse_existing_buildings(dxf_path, scale=args.scale)
        roads = parse_roads(dxf_path, scale=args.scale)
        parcels = parse_cadastral_parcels(dxf_path, scale=args.scale)
        print(f"  Существующие здания: {len(existing_buildings)}")
        print(f"  Дороги: {len(roads)}")
        print(f"  Кадастровые участки: {len(parcels)}")
        if existing_buildings:
            total_existing = sum(b["area_m2"] for b in existing_buildings)
            print(f"  Общая площадь существующей застройки: {total_existing:.0f} м²")

        # Сохраняем в site_data для доступа из других функций
        site_data["existing_buildings"] = existing_buildings
        site_data["roads"] = roads
        site_data["parcels"] = parcels

        num_variants = min(args.variants, 5)
        all_results = []  # для сравнения вариантов

        for variant_idx in range(num_variants):
            variant_seed = 42 + variant_idx * 17
            variant_label = f" (вариант {variant_idx + 1}/{num_variants}, seed={variant_seed})" if num_variants > 1 else ""

            if num_variants > 1:
                print(f"\n{'='*60}")
                print(f"  ВАРИАНТ {variant_idx + 1}/{num_variants} (seed={variant_seed})")
                print(f"{'='*60}")

            # Устанавливаем seed для воспроизводимости варианта
            import random
            random.seed(variant_seed)

            # Парсельная раскладка (приоритет — как у Ильи)
            parcel_buildings = None
            blocks = None
            slots = None
            infill = None

            if site_data.get("parcels"):
                print("\nРаскладка по кадастровым участкам...")
                parcel_buildings = generate_parcel_based_layout(site_data, config)

            if not parcel_buildings or len(parcel_buildings) < 5:
                # Фоллбэк на кварталы
                print("\nФоллбэк: генерация кварталов...")
                blocks = generate_courtyard_blocks(site_data, config)

                internal_roads = generate_internal_roads(blocks, site_data)
                if internal_roads:
                    print(f"  Внутренние проезды: {len(internal_roads)}")
                    site_data["internal_roads"] = internal_roads

                if len(blocks) < 3:
                    print("  Мало кварталов, переключаюсь на слоты...")
                    slots = generate_building_slots(site_data, config)
                else:
                    infill = generate_infill_buildings(site_data, config, blocks)
                    if infill:
                        print(f"  + {len(infill)} infill-зданий в пустых зонах")
                parcel_buildings = None

            # Генерация массинга (AI назначает этажность)
            prev_errors = []
            for attempt in range(1, args.max_retries + 1):
                print(f"\nГенерация массинга (попытка {attempt}/{args.max_retries})...")
                prompt = build_prompt(site_data, config, slots=slots,
                                     blocks=blocks if not slots else None,
                                     infill=infill if not slots else None,
                                     parcel_buildings=parcel_buildings)
                if prev_errors:
                    prompt += "\n\n## ОШИБКИ ПРЕДЫДУЩЕЙ ПОПЫТКИ (исправь их!)\n"
                    for err in prev_errors:
                        prompt += f"- {err}\n"
                    prompt += "\nВСЕ здания ОБЯЗАНЫ находиться ВНУТРИ полигона участка с учётом отступов."
                massing = generate_massing(prompt, args.api_key, args.base_url)

                summary = massing.get("summary", {})
                print(f"  Зданий: {summary.get('total_buildings', '?')}")
                print(f"  Продаваемая площадь: {summary.get('total_sellable_area', '?')} м²")
                print(f"  Плотность: {summary.get('density', '?')}")

                if blocks and not slots:
                    _inject_block_ids(massing, blocks)

                errors = validate_massing(massing, site_data, config)
                if not errors:
                    print("  Валидация: OK")
                    break
                else:
                    print(f"  Ошибки валидации:")
                    for err in errors:
                        print(f"    - {err}")
                    prev_errors = errors
                    if attempt < args.max_retries:
                        print("  Повторная генерация с учётом ошибок...")
            else:
                print("\nНе удалось сгенерировать валидный массинг. Обрезаем и сохраняем.")

            # Пост-обработка
            print("\nПост-обработка...")
            massing = clip_massing_to_buildable(massing, site_data, config)
            summary = massing.get("summary", {})
            print(f"  Финальный результат: {summary.get('total_buildings', 0)} зданий, "
                  f"{summary.get('total_sellable_area', 0)} м² продаваемой площади")

            # Расчёт квартир и паркинга
            total_sell = summary.get("total_sellable_area", 0)
            avg_apartment_m2 = 55
            est_apartments = round(total_sell / avg_apartment_m2) if total_sell else 0
            est_parking = round(est_apartments * 0.7)
            est_residents = round(est_apartments * 2.3)
            summary["est_apartments"] = est_apartments
            summary["est_parking_spots"] = est_parking
            summary["est_residents"] = est_residents
            print(f"  Расчётные квартиры: ~{est_apartments} (ср. {avg_apartment_m2} м²)")
            print(f"  Паркинг: ~{est_parking} м/м | Жители: ~{est_residents} чел.")

            # Проверка инсоляции
            print("\nПроверка инсоляции (22 марта, СанПиН)...")
            insol_violations = check_insolation(massing, config)
            if insol_violations:
                print(f"  Нарушения инсоляции: {len(insol_violations)} зданий")
                for v in insol_violations:
                    print(f"    Здание {v['building_id']} ({v['floors']}эт): "
                          f"макс. непрерывно {v['max_continuous']}ч (нужно {v['required']}ч)")
                print("  Корректировка этажности...")
                fix_insolation_violations(massing, insol_violations, config)
                buildings = massing.get("buildings", [])
                total_gross = sum(b.get("gross_area", 0) for b in buildings)
                total_sell = sum(b.get("sellable_area", 0) for b in buildings)
                massing["summary"]["total_gross_area"] = total_gross
                massing["summary"]["total_sellable_area"] = total_sell
                massing["summary"]["density"] = round(total_gross / site_data["area_m2"], 2)
                insol_v2 = check_insolation(massing, config)
                if insol_v2:
                    print(f"  После корректировки: ещё {len(insol_v2)} нарушений (допустимо для MVP)")
                else:
                    print("  Инсоляция: OK после корректировки")
            else:
                print("  Инсоляция: OK")

            # Визуализация
            suffix = f"_v{variant_idx + 1}" if num_variants > 1 else "_latest"
            viz_path = input_path.parent / "test_output" / f"massing{suffix}.png"
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                visualize_massing(massing, site_data, str(viz_path), config=config)
            except Exception as e:
                print(f"  Визуализация не удалась: {e}")

            # Запись DXF
            dxf_suffix = f"_v{variant_idx + 1}" if num_variants > 1 else ""
            output_dxf = Path(tmpdir) / f"{input_path.stem}_massing{dxf_suffix}.dxf"
            write_massing_to_dxf(massing, site_data, str(output_dxf))

            # Выходной путь
            if args.output:
                base_output = Path(args.output)
                if num_variants > 1:
                    final_output = base_output.parent / f"{base_output.stem}_v{variant_idx + 1}{base_output.suffix}"
                else:
                    final_output = base_output
            else:
                final_output = input_path.parent / f"{input_path.stem}_massing{input_path.suffix}"

            if final_output.suffix.lower() == ".dwg":
                print("Конвертация DXF → DWG...")
                result_dwg = convert_dxf_to_dwg(str(output_dxf), str(final_output.parent), args.oda_path)
                print(f"\nГотово: {result_dwg}")
            else:
                import shutil
                shutil.copy2(str(output_dxf), str(final_output))
                print(f"\nГотово: {final_output}")

            # JSON
            json_output = final_output.with_suffix(".json")
            with open(json_output, "w") as f:
                json.dump(massing, f, indent=2, ensure_ascii=False)
            print(f"JSON: {json_output}")

            # Запоминаем для сравнения
            summary = massing.get("summary", {})
            all_results.append({
                "variant": variant_idx + 1,
                "seed": variant_seed,
                "buildings": summary.get("total_buildings", 0),
                "sellable_area": summary.get("total_sellable_area", 0),
                "density": summary.get("density", 0),
                "apartments": summary.get("est_apartments", 0),
                "parking": summary.get("est_parking_spots", 0),
                "viz": str(viz_path),
            })

        # Сравнительная таблица (если несколько вариантов)
        if num_variants > 1 and all_results:
            print(f"\n{'='*60}")
            print("  СРАВНЕНИЕ ВАРИАНТОВ")
            print(f"{'='*60}")
            print(f"  {'#':<4} {'Зданий':<8} {'Площадь':>10} {'Плотн.':>8} {'Кв-ры':>8} {'Парк.':>8}")
            print(f"  {'-'*4} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
            best = max(all_results, key=lambda r: r["sellable_area"])
            for r in all_results:
                marker = " ★" if r == best else ""
                print(f"  v{r['variant']:<3} {r['buildings']:<8} {r['sellable_area']:>10,} {r['density']:>8} {r['apartments']:>8,} {r['parking']:>8,}{marker}")
            print(f"\n  Лучший: вариант {best['variant']} ({best['viz']})")


if __name__ == "__main__":
    main()
