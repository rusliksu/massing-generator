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


def compute_buildable_area(site_data: dict, config: dict) -> dict:
    """Вычисляет зону застройки с учётом отступов."""
    site_polygon = Polygon(site_data["coordinates"])
    setback = config.get("setbacks", {}).get("default", 6)
    buildable = site_polygon.buffer(-setback)

    # Упрощаем для промпта (меньше вершин)
    simplified = buildable.simplify(2.0, preserve_topology=True)
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


def build_prompt(site_data: dict, config: dict, slots: list[dict] = None) -> str:
    """Формирование промпта для AI."""
    constraints = config.get("constraints", {})
    fire = config.get("fire_safety", {})
    insol = config.get("insolation", {})
    region = config.get("site", {}).get("region", "moscow")
    latitude = config.get("site", {}).get("latitude", 55.75)

    buildable = compute_buildable_area(site_data, config)

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

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

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
    kept = []
    for b in valid_buildings:
        fp = Polygon(b["footprint"])
        conflict = False
        for k in kept:
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


def validate_massing(massing: dict, site_data: dict, config: dict, overlap_threshold: float = 0.85) -> list[str]:
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

    # Проверка противопожарных разрывов
    min_dist = fire.get("min_distance", 12)
    for i in range(len(buildings_polygons)):
        for j in range(i + 1, len(buildings_polygons)):
            dist = buildings_polygons[i].distance(buildings_polygons[j])
            if dist < min_dist:
                errors.append(
                    f"Здания {i+1} и {j+1}: расстояние {dist:.1f}м < мин. {min_dist}м"
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

    # Рисуем границу участка в оригинальных единицах
    site_coords = site_data.get("coordinates_original", site_data["coordinates"])
    msp.add_lwpolyline(
        [(c[0], c[1]) for c in site_coords],
        dxfattribs={"layer": "SITE_BOUNDARY"},
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

        # Генерация слотов для зданий
        print("\nГенерация слотов...")
        slots = generate_building_slots(site_data, config)

        # Генерация массинга
        prev_errors = []
        for attempt in range(1, args.max_retries + 1):
            print(f"\nГенерация массинга (попытка {attempt}/{args.max_retries})...")
            prompt = build_prompt(site_data, config, slots=slots if slots else None)
            if prev_errors:
                prompt += "\n\n## ОШИБКИ ПРЕДЫДУЩЕЙ ПОПЫТКИ (исправь их!)\n"
                for err in prev_errors:
                    prompt += f"- {err}\n"
                prompt += "\nВСЕ здания ОБЯЗАНЫ находиться ВНУТРИ полигона участка с учётом отступов. Используй координаты полигона, а не bounding box."
            massing = generate_massing(prompt, args.api_key, args.base_url)

            summary = massing.get("summary", {})
            print(f"  Зданий: {summary.get('total_buildings', '?')}")
            print(f"  Продаваемая площадь: {summary.get('total_sellable_area', '?')} м²")
            print(f"  Плотность: {summary.get('density', '?')}")

            # Валидация
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

        # Пост-обработка: удалить здания вне зоны
        print("\nПост-обработка...")
        massing = clip_massing_to_buildable(massing, site_data, config)
        summary = massing.get("summary", {})
        print(f"  Финальный результат: {summary.get('total_buildings', 0)} зданий, "
              f"{summary.get('total_sellable_area', 0)} м² продаваемой площади")

        # Запись результата в DXF
        output_dxf = Path(tmpdir) / f"{input_path.stem}_massing.dxf"
        write_massing_to_dxf(massing, site_data, str(output_dxf))

        # Определяем выходной путь
        if args.output:
            final_output = Path(args.output)
        else:
            final_output = input_path.parent / f"{input_path.stem}_massing{input_path.suffix}"

        # Конвертация обратно в DWG если нужно
        if final_output.suffix.lower() == ".dwg":
            print("Конвертация DXF → DWG...")
            result_dwg = convert_dxf_to_dwg(str(output_dxf), str(final_output.parent), args.oda_path)
            print(f"\nГотово: {result_dwg}")
        else:
            import shutil
            shutil.copy2(str(output_dxf), str(final_output))
            print(f"\nГотово: {final_output}")

        # Сохраняем JSON для отладки
        json_output = final_output.with_suffix(".json")
        with open(json_output, "w") as f:
            json.dump(massing, f, indent=2, ensure_ascii=False)
        print(f"JSON: {json_output}")


if __name__ == "__main__":
    main()
