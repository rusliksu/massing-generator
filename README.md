# Massing Generator

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Shapely](https://img.shields.io/badge/Shapely-2.0-green)
![ezdxf](https://img.shields.io/badge/ezdxf-DXF%20I%2FO-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-11557c)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

Automated residential massing generator: **DXF in → buildings out** (DXF + PNG).

Takes a site boundary from a DXF/DWG file, generates optimal building placement using perimeter block layout with Russian building code compliance, and outputs DXF drawings + PNG visualizations.

## How It Works

```
DXF ──► parse boundary ──► perimeter block layout ──► DXF + PNG
              │                      │
         site polygon         150×100m quarters
         buildable zone       sections along perimeter
         main angle           courtyard inside
                              streets between blocks
```

1. **Parse** — extract site boundary from DXF (closed polylines via ezdxf)
2. **Buildable zone** — apply setbacks with mitre buffer
3. **Layout** — place 150×100m perimeter blocks with 18m streets, buildings as sections along edges
4. **Variants** — generate N layouts with different seeds, rank by total area
5. **Output** — DXF with layers (SITE_BOUNDARY, MASSING_BUILDINGS, MASSING_INFO) + PNG visualization

## Features

- **Perimeter block layout** — buildings along edges of rectangular quarters, courtyards inside
- **Russian building codes** — fire safety (SP 4.13130), sanitary distances (SP 42.13330), insolation (SanPiN)
- **Auto unit detection** — handles both mm and m coordinate systems in DXF
- **Seed-based reproducibility** — deterministic layouts via `random.Random(seed)`
- **Multi-variant generation** — generate 20+ variants, pick top-3 by area
- **Summary metrics** — KZ (building coverage), KPZ (floor area ratio), green ratio, apartments, parking, residents

## Norms & Constraints

| Norm | Value | Source |
|------|-------|--------|
| Fire gap (reinforced concrete) | 6 m | SP 4.13130.2013 |
| Street width | 18 m | SP 4.13130 p.8.1 |
| Facade-to-facade (5+ floors) | ≥25 m | SP 42.13330.2016 |
| Insolation | 2h continuous | SanPiN |
| Green space | ≥25% of quarter | SP 42.13330 |
| Parking | ~1 spot/apartment | SP 42.13330 |
| Building coverage (KZ) | 0.2–0.4 | SP 42.13330 |
| Floor area ratio (KPZ) | 1.2–3.0 | SP 42.13330 |

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from massing import *

site = parse_site_boundary("site.dxf")
config = {"setback": 5, "max_floors": 16}

buildings = generate_parcel_based_layout(site, config, seed=42, n_variants=10)
summary = compute_summary(site, config, buildings)

viz = {"buildings": buildings, "summary": summary}
visualize_massing(viz, site, "output.png", config)
write_massing_to_dxf(viz, site, "output.dxf")
```

## Example Output

```
Perimeter layout: 16 floors (48m), block 150×100m, section 50m, street 18m
Placed 37 buildings (best of 10 attempts, seed=9)

Buildings:    37
Apartments:   4,756
KZ:           0.17
KPZ:          2.53
Green space:  85%
```

## Dependencies

- **Python 3.10+**
- `ezdxf` — DXF read/write
- `shapely` — geometry operations
- `numpy` — linear algebra
- `matplotlib` — visualization
- [ODA File Converter](https://www.opendesign.com/guestfiles/oda_file_converter) — optional, for DWG↔DXF

## License

MIT
