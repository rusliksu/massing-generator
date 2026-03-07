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


def parse_site_boundary(dxf_path: str, layer: str = None) -> dict:
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

    bounds = site.bounds  # (minx, miny, maxx, maxy)
    coords = list(site.exterior.coords)

    return {
        "coordinates": coords,
        "area_m2": round(site.area, 2),
        "bounds": {
            "min_x": round(bounds[0], 2),
            "min_y": round(bounds[1], 2),
            "max_x": round(bounds[2], 2),
            "max_y": round(bounds[3], 2),
        },
        "width": round(bounds[2] - bounds[0], 2),
        "height": round(bounds[3] - bounds[1], 2),
        "centroid": {
            "x": round(site.centroid.x, 2),
            "y": round(site.centroid.y, 2),
        },
        "all_polygons_count": len(polygons),
    }


def build_prompt(site_data: dict, config: dict) -> str:
    """Формирование промпта для AI."""
    constraints = config.get("constraints", {})
    setbacks = config.get("setbacks", {})
    fire = config.get("fire_safety", {})
    insol = config.get("insolation", {})
    region = config.get("site", {}).get("region", "moscow")
    latitude = config.get("site", {}).get("latitude", 55.75)

    return f"""Ты — архитектор-градостроитель. Сгенерируй оптимальный массинг (объёмно-пространственное решение) для жилой застройки.

## Пятно застройки
- Координаты полигона: {json.dumps(site_data['coordinates'])}
- Площадь участка: {site_data['area_m2']} м²
- Размеры: {site_data['width']} x {site_data['height']} м
- Центр: ({site_data['centroid']['x']}, {site_data['centroid']['y']})

## Ограничения
- Макс. плотность застройки: {constraints.get('max_density', 2.5)}
- Целевая продаваемая площадь: {constraints.get('target_area', 45000)} м²
- Макс. этажность: {constraints.get('max_floors', 25)}
- Высота этажа: {constraints.get('min_floor_height', 3.0)} м

## Отступы
- От границ участка: {setbacks.get('default', 6)} м
- От дороги: {setbacks.get('road', 10)} м

## Пожарная безопасность
- Мин. расстояние между зданиями: {fire.get('min_distance', 12)} м

## Инсоляция
- Регион: {region}, широта: {latitude}°
- Мин. часы инсоляции: {insol.get('min_hours', 2.0)} ч
- Ориентируй здания для максимальной инсоляции (юг/юго-восток предпочтительнее)

## Нормативная база
Учитывай: СП 42.13330, СанПиН по инсоляции, противопожарные нормы для региона {region}.

## Формат ответа
Верни ТОЛЬКО валидный JSON (без markdown):
{{
  "buildings": [
    {{
      "id": 1,
      "type": "residential",
      "footprint": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "floors": 22,
      "floor_height": 3.0,
      "total_height": 66.0,
      "gross_area": 15000,
      "sellable_area": 12000,
      "orientation_deg": 15
    }}
  ],
  "summary": {{
    "total_buildings": 3,
    "total_gross_area": 50000,
    "total_sellable_area": 45000,
    "density": 2.3,
    "site_coverage_ratio": 0.35,
    "notes": "описание решения"
  }}
}}

Оптимизируй по: максимум продаваемой площади при соблюдении всех норм."""


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
        max_tokens=4096,
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


def validate_massing(massing: dict, site_data: dict, config: dict) -> list[str]:
    """Валидация сгенерированного массинга."""
    errors = []
    constraints = config.get("constraints", {})
    fire = config.get("fire_safety", {})

    site_polygon = Polygon(site_data["coordinates"])
    setback_default = config.get("setbacks", {}).get("default", 6)
    buildable_area = site_polygon.buffer(-setback_default)

    buildings_polygons = []

    for b in massing.get("buildings", []):
        footprint = Polygon(b["footprint"])
        buildings_polygons.append(footprint)

        # Проверка: здание внутри пятна с учётом отступов
        if not buildable_area.contains(footprint):
            errors.append(f"Здание {b['id']}: выходит за пятно застройки (с учётом отступа {setback_default}м)")

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

    # Создаём слои
    doc.layers.add("SITE_BOUNDARY", color=7)       # Белый — граница участка
    doc.layers.add("MASSING_BUILDINGS", color=1)    # Красный — здания
    doc.layers.add("MASSING_INFO", color=3)         # Зелёный — информация

    # Рисуем границу участка
    site_coords = site_data["coordinates"]
    msp.add_lwpolyline(
        [(c[0], c[1]) for c in site_coords],
        dxfattribs={"layer": "SITE_BOUNDARY"},
        close=True,
    )

    # Рисуем здания
    for b in massing.get("buildings", []):
        footprint = b["footprint"]

        # Контур здания
        msp.add_lwpolyline(
            [(p[0], p[1]) for p in footprint],
            dxfattribs={"layer": "MASSING_BUILDINGS"},
            close=True,
        )

        # Подпись: этажность и площадь
        centroid = Polygon(footprint).centroid
        label = f"{b.get('floors', '?')}эт. / {b.get('sellable_area', '?')}м²"
        msp.add_text(
            label,
            dxfattribs={
                "layer": "MASSING_INFO",
                "height": max(site_data["width"], site_data["height"]) / 50,
                "insert": (centroid.x, centroid.y),
            },
        )

    # Сводная информация
    summary = massing.get("summary", {})
    info_y = site_data["bounds"]["min_y"] - site_data["height"] * 0.15
    info_x = site_data["bounds"]["min_x"]
    text_height = max(site_data["width"], site_data["height"]) / 40

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
        site_data = parse_site_boundary(dxf_path, layer=args.layer)
        print(f"  Площадь участка: {site_data['area_m2']} м²")
        print(f"  Размеры: {site_data['width']} x {site_data['height']} м")
        print(f"  Найдено полигонов: {site_data['all_polygons_count']}")

        # Генерация массинга
        for attempt in range(1, args.max_retries + 1):
            print(f"\nГенерация массинга (попытка {attempt}/{args.max_retries})...")
            prompt = build_prompt(site_data, config)
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
                if attempt < args.max_retries:
                    print("  Повторная генерация с учётом ошибок...")
        else:
            print("\nНе удалось сгенерировать валидный массинг. Сохраняем последний результат.")

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
