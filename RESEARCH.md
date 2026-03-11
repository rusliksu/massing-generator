# Ресёрч: автоматизированный массинг (12.03.2026)

## 1. Топ-5 продуктов

- **Autodesk Forma** (бывш. Spacemaker) — облачный 3D-массинг + анализ среды (ветер, шум, инсоляция)
- **TestFit** — constraint-based, parking-first, unit mix, real-time ТЭП. Urban Planner модуль
- **Digital Blue Foam** — массинг + daylight + carbon
- **Archistar** — building envelope + planning approval prediction
- **Modelur** — плагин SketchUp, параметрический урбан-дизайн

## 2. Алгоритмы

- **Rule-based + параметрика** (TestFit, Modelur) — быстро, предсказуемо
- **GA (NSGA-II/III)** (EvoMass) — multi-objective, Pareto-фронт
- **Deep RL** (DRL-urban-planning, Tsinghua) — последовательное размещение
- **Рекомендация:** гибрид rule-based + GA (pymoo)

## 3. Российские нормы

### Пожарные расстояния (СП 4.13130.2013)
- Ж/б ↔ ж/б: **6м**, каркас: **8-10м**, дерево: **12-15м**

### Санитарные разрывы (СП 42.13330.2016)
- Длинная↔длинная 5+эт: **25-45м (~2H)**
- Фасад↔торец: **>=10м**
- Торец↔торец: по пожарным

### Инсоляция (СанПиН)
- Центральная зона (58-48°): **2ч непрерывно**, 22 мар — 22 сен

### Плотность
- КЗ: **0.2-0.4**, КПЗ многоэт: **1.2-3.0**

### Парковки
- **~1 м/м на квартиру**, гостевые <=200м от подъезда

### Озеленение
- **>=25%** площади квартала, **>=16 м²** на человека

## 4. Open-source
- [DRL-urban-planning](https://github.com/tsinghua-fib-lab/DRL-urban-planning) — Python+Shapely, Nature 2023
- [pymoo](https://pymoo.org/) — NSGA-II/III оптимизация
- [EvoMass](https://www.food4rhino.com/en/app/evomass) — GA-массинг

## 5. Архитектура
1. **Constraints** — пожарка, санитарные, инсоляция, граница
2. **Generator** — типологии как атомы, rule-based
3. **Optimizer** — pymoo NSGA-II (площадь↑, затенение↓)
