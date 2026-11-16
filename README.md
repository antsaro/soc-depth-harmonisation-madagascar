# soc-depth-harmonisation-madagascar

Code, tables, figures and manuscript files for:

> *Can a single coefficient extend soil organic carbon stocks from 30 cm to 1 m? Evidence from 748 paired profiles on tropical upland soils.*

The analysis compares eleven depth-harmonisation models that predict the SOC stock of 0–100 cm from the stock of 0–30 cm, selects one, and applies it to 2,664 plots sampled to 30 cm only.

## Selected model

```
SOC100 = 13.45 + 1.618 × SOC30        (Mg C ha-1; upland soils, SOC30 between 5 and 182)
```

Empirical 90 % prediction interval (multiply the prediction): ×0.56–1.91 for SOC30 < 45.4, ×0.70–1.50 for 45.4–79.6, ×0.72–1.32 for SOC30 > 79.6. Not valid for mangrove soils.

## Reproduce

```bash
pip install -r requirements.txt
python src/01_run_analysis.py data/raw/SOCv5.xlsx   # data preparation, cross-validation, tests, predictions (about 1 min)
python src/02_make_figures.py                       # Figures 2-6 and S1-S4
python src/03_study_area_map.py data/raw/SOCv5.xlsx <WDPA_Madagascar.shp>   # Figure 1 (study area + protected areas)
python src/04_mangrove_projects_map.py              # mangrove plots by project (reads the path set at the top of the script)
(cd paper && npm install docx && ./build_all.sh)    # Word files, with native Word (OMML) equations
```

Random seeds are fixed (`SEED = 2026`), so results are reproducible.

## Layout

| Path | Content |
| --- | --- |
| `src/socdepth.py` | Model library: β model, literature coefficients, fitted models, metrics |
| `src/01_run_analysis.py` | Quality control, land-use harmonisation, tests, three cross-validation schemes, predictions |
| `src/02_make_figures.py` | Figures 2–6 and S1–S4 |
| `src/04_mangrove_projects_map.py` | Map of the mangrove sampling plots coloured by project over the 2020 mangrove extent. Input path set at the top: `C:\Users\User\Desktop\LRI\Data_Mangroves_All_Projects_filled_WBconverted_v2.xlsx`, with `gmw_v3_2020_Mada.gpkg` expected in the same folder |
| `src/03_study_area_map.py` | Figure 1: SOC 0–30 and 0–100 cm plots with protected areas within 10 km |
| `paper/omml.py` | Equations as native Word math (OMML), injected into the .docx after building |
| `data/raw/SOCv5.xlsx` | Input compilation (4,184 plots) |
| `data/processed/SOC_Madagascar_harmonised_0_100cm.xlsx` / `.csv` | Harmonised dataset (see its README sheet) |
| `data/processed/calibration_with_cv_predictions.csv` | Out-of-sample predictions of every model for the 748 calibration plots |
| `data/basemap/` | Coastline (GSHHG), Natural Earth places and countries used in the maps |
| `outputs/tables/` | All tables as CSV, plus `summary_statistics.json` |
| `outputs/figures/` | Figures as PNG (400 dpi) and PDF |
| `paper/` | Word manuscript, supplement, highlights and the scripts that build them |

## Land use

Field land-use labels are harmonised into nine detailed classes and merged into four groups (forests and tree plantations; shrublands; grasslands and degraded lands; agroecosystems), plus mangroves. Plots without a field label get their group from ESA WorldCover 2021 v200 (Zanaga et al., 2022), read at the plot coordinates and stored in `data/raw/soc_plots_worldcover2021.csv` (made with `data/raw/worldcover2021_sample_points.js` in Google Earth Engine). The mapping is defined in `src/socdepth.py` (`LULC_GROUP`, `WC_GROUP`).

## Input columns

`Base` = data source (project), `LU` = land use, (added) `KG` = Köppen–Geiger class 1986–2010 (Rubel et al., 2017, via the `kgcpy` package), `Latitude`, `Longitude`, `SC30` = SOC stock 0–30 cm, `SOC100` = SOC stock 0–100 cm (blank if not measured). Stocks are assumed to be in Mg C ha-1.

## Notes

- Protected-area polygons (WDPA, UNEP-WCMC & IUCN) are not redistributed; download them from https://www.protectedplanet.net and pass the path to `src/03_study_area_map.py`.

- The raw data belong to the contributing projects. Check with the data owners before making this repository public.
- Quality-control rules and the set-aside source are defined at the top of section 1 of `src/01_run_analysis.py`.
- Licence: MIT for the code (see `LICENSE`); add a data licence before publication.
