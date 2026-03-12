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
    buildable = site_polygon.buffer(-setback, join_style='mitre', mitre_limit=2.0)

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


def _generate_grid_layout(site_data: dict, config: dict, buildable, main_angle: float,
                           seed: int = None, n_variants: int = 30) -> list[dict]:
    """Квартальная раскладка: разбивает участок на блоки с проездами,
    внутри каждого блока размещает 2-4 здания параллельными рядами.
    seed — для воспроизводимости."""
    import numpy as np
    from shapely.geometry import box, Point, LineString
    from shapely.affinity import rotate, translate
    import random

    rng = random.Random(seed)
    area = buildable.area

    # Для маленьких участков (<1 га) — простая раскладка без кварталов
    if area < 10000:
        return _generate_small_site_layout(config, buildable, main_angle, rng)

    # === Периметральная застройка ===
    #
    #   Квартал ~150×100м. Здания — секциями по периметру, двор внутри.
    #   Между кварталами — улицы 15-20м (пожарный проезд + тротуары).
    #
    #   ┌──────┐  ┌──────┐  ┌──────┐     ← секции по длинной стороне
    #   │      │  │      │  │      │
    #   │  ┌───┘  └──────┘  └───┐  │
    #   │  │                    │  │     ← торцевые секции
    #   │  │       ДВОР         │  │
    #   │  │                    │  │
    #   │  └───┐  ┌──────┐  ┌──┘  │
    #   │      │  │      │  │     │
    #   └──────┘  └──────┘  └─────┘
    #

    # Автодетект единиц: если площадь > 10^8, координаты в мм → масштаб 1000
    U = 1000.0 if area > 1e8 else 1.0
    if U > 1:
        print(f"  Координаты в мм (area={area:.0f}), масштаб ×{U:.0f}")

    DEPTH = 13 * U        # глубина секции (м→ед.чертежа)
    FLOOR_H = 3.0         # высота этажа (м) — не масштабируется
    FIRE_GAP = 6 * U      # ж/б ↔ ж/б (СП 4.13130, I-II степень)
    STREET_W = 18 * U     # улица между кварталами (проезд+тротуар)
    SECTION_LEN = 50 * U  # длина одной секции
    SECTION_GAP = 6 * U   # разрыв между секциями (проход во двор)
    BLOCK_W = 150 * U     # квартал вдоль main_angle
    BLOCK_H = 100 * U     # квартал поперёк

    target_floors = config.get("max_floors", 16)
    building_h = target_floors * FLOOR_H

    print(f"  Периметральная застройка: {target_floors} эт. ({building_h:.0f}м), "
          f"блок {BLOCK_W/U:.0f}×{BLOCK_H/U:.0f}м, секция {SECTION_LEN/U:.0f}м, "
          f"улица {STREET_W/U:.0f}м, разрыв {SECTION_GAP/U:.0f}м")

    rad = np.radians(main_angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    cos_p, sin_p = -sin_a, cos_a  # перпендикулярное направление

    cx, cy = buildable.centroid.x, buildable.centroid.y
    span = area ** 0.5 * 1.2

    step_long = BLOCK_W + STREET_W
    step_across = BLOCK_H + STREET_W

    n_long = int(span / step_long) + 1
    n_across = int(span / step_across) + 1

    def _place_section(ox, oy, angle, length):
        """Создаёт одну секцию здания (прямоугольник) в заданной точке и ориентации."""
        hw, hd = length / 2, DEPTH / 2
        fp = box(ox - hw, oy - hd, ox + hw, oy + hd)
        fp_rot = rotate(fp, angle, origin=(ox, oy))
        is_long = abs((angle - main_angle) % 180) < 10 or abs((angle - main_angle) % 180) > 170
        return {
            'poly': fp_rot, 'cx': ox, 'cy': oy,
            'coords': list(fp_rot.exterior.coords)[:-1],
            'area': fp_rot.area, 'len': length,
            'type': 'long' if is_long else 'short',
            'angle': angle,
        }

    def _make_perimeter_block(qx, qy):
        """Создаёт периметральный квартал: секции по 4 сторонам прямоугольника."""
        sections = []
        half_w = BLOCK_W / 2
        half_h = BLOCK_H / 2

        # --- Длинные стороны (вдоль main_angle) ---
        # Сколько секций помещается по длинной стороне
        usable_long = BLOCK_W - DEPTH  # за вычетом углов (торцевые секции)
        n_sec = max(1, int((usable_long + SECTION_GAP) / (SECTION_LEN + SECTION_GAP)))
        actual_gap = (usable_long - n_sec * SECTION_LEN) / max(1, n_sec - 1) if n_sec > 1 else 0
        actual_gap = max(actual_gap, SECTION_GAP)
        # Пересчитать длину секции если зазоры вышли слишком большие
        sec_len = min(SECTION_LEN, (usable_long - (n_sec - 1) * actual_gap) / n_sec) if n_sec > 0 else SECTION_LEN

        for side in [-1, 1]:  # верхняя и нижняя стороны
            for si in range(n_sec):
                # Смещение секции вдоль длинной стороны
                total_span = n_sec * sec_len + (n_sec - 1) * actual_gap
                start = -total_span / 2 + si * (sec_len + actual_gap) + sec_len / 2
                sx = qx + start * cos_a + side * (half_h - DEPTH / 2) * cos_p
                sy = qy + start * sin_a + side * (half_h - DEPTH / 2) * sin_p
                sections.append(_place_section(sx, sy, main_angle, sec_len))

        # --- Короткие стороны (перпендикулярно main_angle) ---
        usable_short = BLOCK_H - DEPTH
        n_sec_s = max(1, int((usable_short + SECTION_GAP) / (SECTION_LEN + SECTION_GAP)))
        # Торцевые секции могут быть короче
        sec_len_s = min(SECTION_LEN, (usable_short - (n_sec_s - 1) * SECTION_GAP) / n_sec_s) if n_sec_s > 0 else SECTION_LEN

        for end in [-1, 1]:  # левая и правая стороны
            for si in range(n_sec_s):
                total_span_s = n_sec_s * sec_len_s + (n_sec_s - 1) * SECTION_GAP
                start = -total_span_s / 2 + si * (sec_len_s + SECTION_GAP) + sec_len_s / 2
                sx = qx + end * (half_w - DEPTH / 2) * cos_a + start * cos_p
                sy = qy + end * (half_w - DEPTH / 2) * sin_a + start * sin_p
                sections.append(_place_section(sx, sy, main_angle + 90, sec_len_s))

        return sections

    best_buildings = []
    best_total = 0

    from shapely import prepared
    prep_buildable = prepared.prep(buildable)

    # STREET_W (18м) > FIRE_GAP (6м) → секции разных блоков не могут
    # нарушить пожарные расстояния. Проверяем только containment.

    for attempt in range(min(n_variants, 10)):
        offset_long = rng.uniform(-step_long / 2, step_long / 2)
        offset_across = rng.uniform(-step_across / 2, step_across / 2)

        placed = []

        for bi in range(-n_long, n_long + 1):
            for bj in range(-n_across, n_across + 1):
                qx = cx + (bi * step_long + offset_long) * cos_a \
                         + (bj * step_across + offset_across) * cos_p
                qy = cy + (bi * step_long + offset_long) * sin_a \
                         + (bj * step_across + offset_across) * sin_p

                # Быстрая отсечка: центр квартала внутри зоны?
                if not prep_buildable.contains(Point(qx, qy)):
                    continue

                block = _make_perimeter_block(qx, qy)
                for sec in block:
                    if prep_buildable.contains(sec['poly']):
                        placed.append(sec)

        total = sum(o['area'] for o in placed)
        if total > best_total:
            best_total = total
            best_buildings = placed

    # Финализация
    buildings = _finalize_buildings(best_buildings, rng, target_floors, unit_scale=U)
    print(f"  Размещено {len(buildings)} зданий (периметральная застройка, "
          f"лучшая из {n_variants} попыток, seed={seed})")
    return buildings


def _generate_small_site_layout(config: dict, buildable, main_angle: float,
                                 rng) -> list[dict]:
    """Раскладка для маленьких участков (<1 га): перебор пар/троек."""
    import numpy as np
    from shapely.geometry import box, Point
    from shapely.affinity import rotate

    cx, cy = buildable.centroid.x, buildable.centroid.y
    area = buildable.area
    step = max(6, min(10, area ** 0.5 / 8))
    DEPTH = 13
    FIRE_GAP = 12

    # Кандидатные позиции
    candidates = []
    span = area ** 0.5
    for dx in np.arange(-span, span + 1, step):
        for dy in np.arange(-span, span + 1, step):
            rad = np.radians(main_angle)
            x = cx + dx * np.cos(rad) - dy * np.sin(rad)
            y = cy + dx * np.sin(rad) + dy * np.cos(rad)
            if buildable.contains(Point(x, y)):
                candidates.append((x, y))

    # Все возможные здания
    all_options = []
    for x, y in candidates:
        for angle_offset in [0, 90]:
            angle = main_angle + angle_offset
            for b_len in [80, 70, 60, 50, 45, 40, 35, 30, 25, 20]:
                hw, hd = b_len / 2, DEPTH / 2
                fp = box(x - hw, y - hd, x + hw, y + hd)
                fp_rot = rotate(fp, angle, origin=(x, y))
                if buildable.contains(fp_rot):
                    all_options.append({
                        'poly': fp_rot,
                        'coords': list(fp_rot.exterior.coords)[:-1],
                        'area': fp_rot.area,
                        'cx': x, 'cy': y,
                        'len': b_len,
                    })
                    break

    # Перебор: лучшее одиночное → лучшая пара → лучшая тройка
    best = []
    best_area = 0

    for opt in all_options:
        if opt['area'] > best_area:
            best_area = opt['area']
            best = [opt]

    if len(all_options) <= 150:
        for i in range(len(all_options)):
            for j in range(i + 1, len(all_options)):
                a, b = all_options[i], all_options[j]
                if a['poly'].distance(b['poly']) < FIRE_GAP:
                    continue
                total = a['area'] + b['area']
                if total > best_area:
                    best_area = total
                    best = [a, b]

        if len(best) == 2:
            for k in range(len(all_options)):
                c = all_options[k]
                if any(c['poly'].distance(b['poly']) < FIRE_GAP for b in best):
                    continue
                total = best_area + c['area']
                if total > best_area:
                    best_area = total
                    best = best + [c]

    buildings = _finalize_buildings(best, rng)
    print(f"  Размещено {len(buildings)} зданий (малый участок)")
    return buildings


def _finalize_buildings(placed: list[dict], rng, target_floors: int = 16,
                        unit_scale: float = 1.0) -> list[dict]:
    """Преобразует внутренний формат зданий в выходной.
    Этажность: длинные корпуса — target_floors, торцевые — меньше.
    unit_scale: множитель единиц (1000 если координаты в мм)."""
    buildings = []
    area_div = unit_scale ** 2  # мм² → м²
    for i, b in enumerate(placed, 1):
        fp_area = b['area'] / area_div
        btype = b.get('type', 'long')
        if btype == 'long':
            floors = target_floors
        else:
            # Торцевые обычно ниже (стилобат или пониженный корпус)
            floors = max(5, target_floors - rng.randint(2, 5))
        buildings.append({
            'id': i,
            'footprint': [[round(c[0], 1), round(c[1], 1)] for c in b['coords']],
            'area_m2': round(fp_area, 0),
            'floors': floors,
            'shape': 'rect',
            'cx': round(b['cx'], 1),
            'cy': round(b['cy'], 1),
        })
    return buildings


def compute_summary(site_data: dict, config: dict, buildings: list[dict]) -> dict:
    """Вычисляет сводку: площади, квартиры, машиноместа, жители."""
    raw_area = site_data.get("area_m2", 0)
    # Автодетект единиц: area_m2 > 10^8 → координаты в мм, площадь в мм²
    area_div = 1e6 if raw_area > 1e8 else 1.0
    site_area_m2 = raw_area / area_div

    # area_m2 в зданиях уже конвертировано в м² через _finalize_buildings
    total_footprint = sum(b.get('area_m2', 0) for b in buildings)
    total_sellable = 0
    total_apartments = 0

    for b in buildings:
        fp_area = b.get('area_m2', 0)
        floors = b.get('floors', 10)
        gross = fp_area * floors
        sellable = gross * 0.75  # ~75% продаваемая (минус МОП, лестницы, стены)
        apartments = int(sellable / 55)  # средняя квартира ~55 м²
        total_sellable += sellable
        total_apartments += apartments

    total_residents = int(total_apartments * 2.3)
    parking = total_apartments  # 1 м/м на квартиру (СП 42.13330)
    coverage = total_footprint / site_area_m2 if site_area_m2 > 0 else 0

    # Озеленение: СП 42.13330 — не менее 25% площади квартала
    green_area = site_area_m2 - total_footprint  # грубая оценка (без дорог/парковок)
    green_ratio = green_area / site_area_m2 if site_area_m2 > 0 else 0
    green_ok = green_ratio >= 0.25

    # КПЗ = суммарная поэтажная площадь / площадь участка
    total_gross = sum(b.get('area_m2', 0) * b.get('floors', 1) for b in buildings)
    kpz = total_gross / site_area_m2 if site_area_m2 > 0 else 0

    return {
        'total_buildings': len(buildings),
        'total_sellable_area': f'{total_sellable:.0f}',
        'total_footprint': f'{total_footprint:.0f}',
        'total_area_m2': round(total_gross, 0),
        'site_area_m2': round(site_area_m2, 0),
        'density': f'{coverage:.1%}',
        'kz': round(coverage, 3),
        'site_coverage_ratio': f'{coverage:.1%}',
        'est_apartments': total_apartments,
        'est_parking_spots': parking,
        'est_residents': total_residents,
        'kpz': round(kpz, 2),
        'green_ratio': round(green_ratio, 3),
        'green_ok': green_ok,
    }


def generate_variants(site_data: dict, config: dict, n_seeds: int = 5,
                      n_variants_per_seed: int = 30) -> list[dict]:
    """Генерирует несколько вариантов раскладки с разными seed'ами.
    Возвращает список: [{'seed': int, 'buildings': [...], 'summary': {...}, 'total_area': float}]
    отсортированный по total_area (лучший первый)."""
    results = []
    for s in range(n_seeds):
        buildings = generate_parcel_based_layout(
            site_data, config, seed=s, n_variants=n_variants_per_seed)
        summary = compute_summary(site_data, config, buildings)
        total_area = sum(b.get('area_m2', 0) * b.get('floors', 1) for b in buildings)
        results.append({
            'seed': s,
            'buildings': buildings,
            'summary': summary,
            'total_area': total_area,
        })
    results.sort(key=lambda r: r['total_area'], reverse=True)
    print(f"  Сгенерировано {n_seeds} вариантов. Лучший: seed={results[0]['seed']}, "
          f"площадь={results[0]['total_area']:.0f}м²")
    return results


def generate_parcel_based_layout(site_data: dict, config: dict,
                                  seed: int = None, n_variants: int = 30) -> list[dict]:
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

    if not parcels:
        print("  Нет кадастровых участков — grid-фоллбэк")
        return _generate_grid_layout(site_data, config, buildable, main_angle,
                                     seed=seed, n_variants=n_variants)

    # Собираем полигоны существующих зданий для проверки зазора
    existing_polys = [eb["polygon"] for eb in site_data.get("existing_buildings", [])]

    # Собираем дороги для проверки пересечений
    road_polys = site_data.get("roads", [])

    buildings = []
    building_id = 1

    for parcel in parcels:
        pp = parcel["polygon"]
        area = parcel["area_m2"]

        # Пропускаем слишком маленькие участки
        if area < 300:
            continue

        # Участок должен хотя бы частично пересекаться с зоной застройки
        parcel_in_buildable = buildable.intersection(pp).area / pp.area if pp.area > 0 else 0
        if parcel_in_buildable < 0.15:
            continue

        cx, cy = pp.centroid.x, pp.centroid.y

        # Определяем пропорции участка через MRR
        p_mrr = pp.minimum_rotated_rectangle
        p_coords = list(p_mrr.exterior.coords)
        pe1 = np.linalg.norm(np.array(p_coords[1]) - np.array(p_coords[0]))
        pe2 = np.linalg.norm(np.array(p_coords[2]) - np.array(p_coords[1]))
        p_long = max(pe1, pe2)
        p_short = min(pe1, pe2)

        # Отступ от границ участка (5м — минимальный от границы)
        setback = 5
        avail_long = p_long - setback * 2
        avail_short = p_short - setback * 2

        if avail_long < 20 or avail_short < 10:
            continue  # слишком маленький для здания

        # Глубина корпуса 12-14м (секционный жилой дом, как у Ильи: avg 12м)
        depth = random.uniform(12, 14)
        if depth > avail_short:
            depth = max(10, avail_short)

        # Большие участки — несколько зданий вдоль длинной оси
        # На широких участках первое здание может быть H/L
        if area > 2500 and avail_long > 55:
            num_buildings = min(max(2, int(avail_long / 60)), 4)
            if num_buildings >= 2:
                # Определяем направление длинной оси участка
                p_mrr2 = pp.minimum_rotated_rectangle
                pc2 = list(p_mrr2.exterior.coords)
                pe1v = np.array(pc2[1]) - np.array(pc2[0])
                pe2v = np.array(pc2[2]) - np.array(pc2[1])
                if np.linalg.norm(pe1v) > np.linalg.norm(pe2v):
                    axis = pe1v / np.linalg.norm(pe1v)
                else:
                    axis = pe2v / np.linalg.norm(pe2v)

                # Размещаем здания вдоль оси с шагом
                spacing = avail_long / num_buildings
                start_offset = -(num_buildings - 1) * spacing / 2

                multi_placed = 0
                for k in range(num_buildings):
                    offset = start_offset + k * spacing
                    bx = cx + axis[0] * offset
                    by = cy + axis[1] * offset

                    # Первое здание на широком участке (short > 40м) — может быть H/L
                    is_wide = avail_short > 40
                    if k == 0 and is_wide and random.random() < 0.5:
                        sub_shape = random.choices(["h_shape", "l_shape"], weights=[50, 50], k=1)[0]
                        if sub_shape == "h_shape":
                            sub_len = min(spacing * 0.85, 80)
                            sub_depth = min(60, max(40, sub_len * random.uniform(0.6, 0.85)))
                        else:
                            sub_len = min(spacing * 0.85, 55)
                            sub_depth = min(45, max(30, sub_len * random.uniform(0.7, 0.95)))
                    else:
                        sub_shape = "rect"
                        sub_len = min(spacing * 0.80, 100)
                        sub_len = max(40, sub_len)
                        sub_depth = random.uniform(12, 14)
                    sub_hw, sub_hd = sub_len / 2, sub_depth / 2

                    if sub_shape == "h_shape" and sub_len > 35:
                        wing_r = random.uniform(0.30, 0.42)
                        wing_l = sub_len * wing_r
                        conn_d2 = sub_depth * random.uniform(0.30, 0.50)
                        sub_fp = Polygon([
                            (bx - sub_hw, by - sub_hd),
                            (bx - sub_hw + wing_l, by - sub_hd),
                            (bx - sub_hw + wing_l, by - conn_d2 / 2),
                            (bx + sub_hw - wing_l, by - conn_d2 / 2),
                            (bx + sub_hw - wing_l, by - sub_hd),
                            (bx + sub_hw, by - sub_hd),
                            (bx + sub_hw, by + sub_hd),
                            (bx + sub_hw - wing_l, by + sub_hd),
                            (bx + sub_hw - wing_l, by + conn_d2 / 2),
                            (bx - sub_hw + wing_l, by + conn_d2 / 2),
                            (bx - sub_hw + wing_l, by + sub_hd),
                            (bx - sub_hw, by + sub_hd),
                        ])
                    elif sub_shape == "u_shape" and sub_len > 35:
                        wing_d2 = sub_depth
                        base_d2 = sub_depth * random.uniform(0.35, 0.50)
                        sub_fp = Polygon([
                            (bx - sub_hw, by - sub_hd),
                            (bx + sub_hw, by - sub_hd),
                            (bx + sub_hw, by + sub_hd),
                            (bx + sub_hw - wing_d2, by + sub_hd),
                            (bx + sub_hw - wing_d2, by - sub_hd + base_d2),
                            (bx - sub_hw + wing_d2, by - sub_hd + base_d2),
                            (bx - sub_hw + wing_d2, by + sub_hd),
                            (bx - sub_hw, by + sub_hd),
                        ])
                    elif sub_shape == "l_shape" and sub_len > 30:
                        wr = random.uniform(0.35, 0.55)
                        wdr = random.uniform(0.30, 0.50)
                        sub_fp = Polygon([
                            (bx - sub_hw, by - sub_hd),
                            (bx - sub_hw + sub_len * wr, by - sub_hd),
                            (bx - sub_hw + sub_len * wr, by - sub_hd + sub_depth * wdr),
                            (bx + sub_hw, by - sub_hd + sub_depth * wdr),
                            (bx + sub_hw, by + sub_hd),
                            (bx - sub_hw, by + sub_hd),
                        ])
                    else:
                        sub_fp = box(bx - sub_hw, by - sub_hd, bx + sub_hw, by + sub_hd)

                    # Поворот по оси участка (с 40% шансом перпендикулярно)
                    sub_angle = main_angle + (90 if random.random() < 0.4 else 0)
                    sub_fp_rot = rotate(sub_fp, sub_angle, origin=(bx, by))

                    # Фикс невалидной геометрии
                    if not sub_fp_rot.is_valid:
                        sub_fp_rot = sub_fp_rot.buffer(0)
                    if sub_fp_rot.area == 0:
                        continue
                    # Здание должно полностью влезать в зону застройки
                    if not buildable.contains(sub_fp_rot):
                        continue
                    # Проверка: пожарные разрывы 12м
                    ok = True
                    for existing in buildings:
                        if Polygon(existing["footprint"]).distance(sub_fp_rot) < 12:
                            ok = False
                            break
                    if not ok:
                        continue

                    coords = list(sub_fp_rot.exterior.coords)[:-1]
                    sub_area = sub_fp_rot.area
                    if sub_area > 1000:
                        sf = random.randint(12, 20)
                    elif sub_area > 500:
                        sf = random.randint(9, 16)
                    else:
                        sf = random.randint(5, 12)
                    buildings.append({
                        "id": building_id,
                        "footprint": [[round(c[0], 1), round(c[1], 1)] for c in coords],
                        "area_m2": round(sub_area, 0),
                        "floors": sf,
                        "shape": sub_shape,
                        "parcel_area": round(area / num_buildings, 0),
                        "orientation_deg": round(sub_angle, 1),
                        "cx": round(bx, 1),
                        "cy": round(by, 1),
                    })
                    building_id += 1
                    multi_placed += 1

                if multi_placed > 0:
                    continue  # уже разместили, не ставить одиночное

        # Выбираем форму (как у Ильи: rect 67%, L 12%, H 8%, complex 12%)
        # H и L только если участок достаточно широкий (short > 25м)
        if avail_short > 25 and area > 3000:
            shape_type = random.choices(["rect", "l_shape", "h_shape"], weights=[50, 30, 20], k=1)[0]
        elif avail_short > 20 and area > 2000:
            shape_type = random.choices(["rect", "l_shape"], weights=[65, 35], k=1)[0]
        else:
            shape_type = "rect"

        # Длина здания — заполняем 75-95% доступной длины (у Ильи avg 77м, до 122м)
        b_len = avail_long * random.uniform(0.75, 0.95)
        b_len = max(40, min(b_len, 125))  # 40-125м

        hw, hd = b_len / 2, depth / 2

        if shape_type == "h_shape":
            # Н-образное (как у Ильи: 73x57м, fill ~54%)
            # Ограничиваем размеры до масштаба Ильи
            b_len = min(b_len, 80)
            hw = b_len / 2
            depth = min(60, max(40, b_len * random.uniform(0.6, 0.85)))
            hd = depth / 2
            wing_ratio = random.uniform(0.35, 0.45)
            wing_len = b_len * wing_ratio
            conn_d = depth * random.uniform(0.20, 0.35)

            fp = Polygon([
                (cx - hw, cy - hd),
                (cx - hw + wing_len, cy - hd),
                (cx - hw + wing_len, cy - conn_d / 2),
                (cx + hw - wing_len, cy - conn_d / 2),
                (cx + hw - wing_len, cy - hd),
                (cx + hw, cy - hd),
                (cx + hw, cy + hd),
                (cx + hw - wing_len, cy + hd),
                (cx + hw - wing_len, cy + conn_d / 2),
                (cx - hw + wing_len, cy + conn_d / 2),
                (cx - hw + wing_len, cy + hd),
                (cx - hw, cy + hd),
            ])

        elif shape_type == "u_shape":
            # П-образное
            b_len = min(b_len, 65)
            hw = b_len / 2
            depth = min(50, max(35, b_len * random.uniform(0.6, 0.8)))
            hd = depth / 2
            wing_d = depth
            base_d = depth * random.uniform(0.35, 0.50)
            gap_w = b_len * random.uniform(0.30, 0.45)  # ширина двора

            fp = Polygon([
                (cx - hw, cy - hd),
                (cx + hw, cy - hd),
                (cx + hw, cy + hd),
                (cx + hw - wing_d, cy + hd),
                (cx + hw - wing_d, cy - hd + base_d),
                (cx - hw + wing_d, cy - hd + base_d),
                (cx - hw + wing_d, cy + hd),
                (cx - hw, cy + hd),
            ])

        elif shape_type == "l_shape":
            # Г-образное (как у Ильи: 43x38м, fill ~65%)
            b_len = min(b_len, 55)
            hw = b_len / 2
            depth = min(45, max(30, b_len * random.uniform(0.7, 0.95)))
            hd = depth / 2
            wing_ratio = random.uniform(0.45, 0.60)
            wing_short = b_len * wing_ratio
            wing_depth_ratio = random.uniform(0.40, 0.55)
            fp = Polygon([
                (cx - hw, cy - hd),
                (cx - hw + wing_short, cy - hd),
                (cx - hw + wing_short, cy - hd + depth * wing_depth_ratio),
                (cx + hw, cy - hd + depth * wing_depth_ratio),
                (cx + hw, cy + hd),
                (cx - hw, cy + hd),
            ])

        else:
            # Прямоугольник (вытянутый 1:4 - 1:6)
            fp = box(cx - hw, cy - hd, cx + hw, cy + hd)

        # Поворот по оси конкретного участка (у Ильи два направления ~38° и ~128°)
        p_mrr_r = pp.minimum_rotated_rectangle
        p_coords_r = list(p_mrr_r.exterior.coords)
        pe1v_r = np.array(p_coords_r[1]) - np.array(p_coords_r[0])
        pe2v_r = np.array(p_coords_r[2]) - np.array(p_coords_r[1])
        if np.linalg.norm(pe1v_r) > np.linalg.norm(pe2v_r):
            parcel_angle = np.degrees(np.arctan2(pe1v_r[1], pe1v_r[0]))
        else:
            parcel_angle = np.degrees(np.arctan2(pe2v_r[1], pe2v_r[0]))
        # 40% шанс перпендикулярного направления (как у Ильи: ~40% зданий повёрнуты на 90°)
        if random.random() < 0.4:
            parcel_angle += 90
        fp_rotated = rotate(fp, parcel_angle, origin=(cx, cy))

        # Фикс невалидной геометрии
        if not fp_rotated.is_valid:
            fp_rotated = fp_rotated.buffer(0)

        # Здание должно полностью влезать в зону застройки
        if not buildable.contains(fp_rotated):
            continue

        # Проверка противопожарных разрывов между новыми зданиями (12м)
        skip = False
        for existing in buildings:
            ep = Polygon(existing["footprint"])
            if fp_rotated.distance(ep) < 12:
                skip = True
                break
        if skip:
            continue

        coords = list(fp_rotated.exterior.coords)[:-1]
        # Дефолтная этажность по площади (AI потом уточнит)
        fp_area = fp_rotated.area
        if fp_area > 2000:
            def_floors = random.randint(16, 25)
        elif fp_area > 1000:
            def_floors = random.randint(12, 20)
        elif fp_area > 500:
            def_floors = random.randint(9, 16)
        else:
            def_floors = random.randint(5, 12)
        buildings.append({
            "id": building_id,
            "footprint": [[round(c[0], 1), round(c[1], 1)] for c in coords],
            "area_m2": round(fp_area, 0),
            "floors": def_floors,
            "shape": shape_type,
            "parcel_area": round(area, 0),
            "orientation_deg": round(parcel_angle, 1),
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
    import matplotlib.patheffects
    from matplotlib.collections import PatchCollection

    # Тёмная тема
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    fig.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # Граница участка (яркий голубой)
    site_poly = Polygon(site_data["coordinates"])
    x, y = site_poly.exterior.xy
    ax.plot(x, y, color='#00d4ff', linewidth=2.5, label='Граница участка')

    # Buildable area
    from shapely.geometry import Polygon as ShapelyPolygon
    buildable_info = compute_buildable_area(site_data, config or {})
    buildable = buildable_info["polygon"]
    if buildable.geom_type == 'Polygon':
        bx, by = buildable.exterior.xy
        ax.plot(bx, by, color='#4ecca3', linewidth=1, linestyle='--', alpha=0.6, label='Зона застройки')
    elif buildable.geom_type == 'MultiPolygon':
        for geom in buildable.geoms:
            bx, by = geom.exterior.xy
            ax.plot(bx, by, color='#4ecca3', linewidth=1, linestyle='--', alpha=0.6)

    # Существующие здания
    for eb in site_data.get("existing_buildings", []):
        ex, ey = eb["polygon"].exterior.xy
        ax.fill(ex, ey, facecolor='#555555', edgecolor='#888888', linewidth=1, alpha=0.7)
    if site_data.get("existing_buildings"):
        ax.plot([], [], color='#888888', linewidth=5, alpha=0.7, label=f'Существующие ({len(site_data["existing_buildings"])})')

    # Кадастровые участки
    for parcel in site_data.get("parcels", []):
        px, py = parcel["polygon"].exterior.xy
        ax.plot(px, py, color='#4ecca3', linewidth=0.5, linestyle=':', alpha=0.3)
    if site_data.get("parcels"):
        ax.plot([], [], color='#4ecca3', linewidth=1, linestyle=':', alpha=0.4,
                label=f'Кадастр ({len(site_data["parcels"])})')

    # Внутренние проезды
    for iroad in site_data.get("internal_roads", []):
        road_poly = iroad["polygon"]
        polys = [road_poly] if road_poly.geom_type == 'Polygon' else list(road_poly.geoms)
        for rp in polys:
            irx, iry = rp.exterior.xy
            ax.fill(irx, iry, facecolor='#2a2a3a', edgecolor='#555', linewidth=0.5, alpha=0.6)
    if site_data.get("internal_roads"):
        ax.plot([], [], color='#555', linewidth=5, alpha=0.6,
                label=f'Проезды ({len(site_data["internal_roads"])})')

    # Дороги из DWG
    for road_poly in site_data.get("roads", []):
        polys = [road_poly] if road_poly.geom_type == 'Polygon' else list(road_poly.geoms)
        for rp in polys:
            rx, ry = rp.exterior.xy
            ax.fill(rx, ry, facecolor='#2a2a3a', edgecolor='#555', linewidth=0.5, alpha=0.7)
    if site_data.get("roads"):
        ax.plot([], [], color='#555', linewidth=5, alpha=0.7,
                label=f'Дороги ({len(site_data["roads"])})')

    # Здания (яркие на тёмном фоне)
    building_colors = ['#e94560', '#f39c12', '#00b894', '#6c5ce7',
                       '#fd79a8', '#0984e3', '#00cec9', '#e17055',
                       '#a29bfe', '#55efc4', '#fab1a0', '#74b9ff']
    for i, b in enumerate(massing.get("buildings", [])):
        fp = b["footprint"]
        color = building_colors[i % len(building_colors)]

        ax.add_patch(plt.Polygon(fp, closed=True, facecolor=color,
                                  edgecolor='white', linewidth=2, alpha=0.85))

        # Подпись
        centroid = Polygon(fp).centroid
        floors = b.get("floors", "?")
        sun_h = b.get("max_continuous_sun")
        label = f"{floors}эт"
        if sun_h is not None and sun_h < 2.0:
            label += f"\n⚠{sun_h}ч"
            ax.add_patch(plt.Polygon(fp, closed=True, facecolor='none',
                                      edgecolor='#ff0000', linewidth=3, alpha=0.9))
        ax.text(centroid.x, centroid.y, label, ha='center', va='center',
                fontsize=7, fontweight='bold', color='white',
                path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])

    ax.set_aspect('equal')

    summary = massing.get("summary", {})
    ax.set_title(f"Массинг — {summary.get('total_buildings', '?')} зданий, "
                 f"{summary.get('total_sellable_area', '?')} м²",
                 fontsize=14, color='white', fontweight='bold')
    legend = ax.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                       edgecolor='#444', labelcolor='white')
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(True, alpha=0.15, color='#555')

    # Сводка
    info_parts = [
        f"КЗ: {summary.get('density', '?')}",
        f"КПЗ: {summary.get('kpz', '?')}",
    ]
    if summary.get("est_apartments"):
        info_parts.append(f"~{summary['est_apartments']} кв.")
        info_parts.append(f"~{summary.get('est_parking_spots', '?')} м/м")
        info_parts.append(f"~{summary.get('est_residents', '?')} жит.")
    green = summary.get('green_ratio', '?')
    green_mark = "OK" if summary.get('green_ok') else "МАЛО"
    info_parts.append(f"Озел: {green} ({green_mark})")
    fig.text(0.5, 0.01, " | ".join(info_parts), ha='center', fontsize=9, color='#aaa')

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
