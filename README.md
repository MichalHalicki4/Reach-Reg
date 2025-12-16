# 🌊 Reach-Reg: Daily Water Surface Elevation from Multi-Mission Satellite Altimetry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxx.svg)](https://doi.org/10.5281/zenodo.17928117) 

## 📝 Overview

The **Reach-Reg** method is a robust, computationally efficient, and open-source approach designed to reconstruct **daily river Water Surface Elevation ($\text{WSE}$) time series** from multi-mission satellite altimetry observations (SWOT, Sentinel-3A/B, Sentinel-6) from the Dahiti database (https://dahiti.dgfi.tum.de/en/map). The method is also applicable to the RiverSP product (SWOT-only) using the Hydrocon API (https://podaac.github.io/hydrocron/hydrocron_tutorial.html). The currently processed daily WSE can be accessed in our Zenodo repository: https://doi.org/10.5281/zenodo.17928117

By leveraging the spatial density of SWOT data to establish empirical, time-invariant, linear relationships between neighboring river reaches, Reach-Reg effectively converts sparse, multi-mission observations into dense, daily time series. The resulting high-frequency $\text{WSE}$ products are crucial for hydrological modeling and monitoring in data-sparse regions.

### Key Features
* **Multi-Mission Fusion:** Integrates data from various satellite altimetry missions.
* **Spatio-Temporal Regression:** Uses Linear Regression (LR) and Orthogonal Distance Regression (ODR) fitting to establish hydraulic relationships between reaches.
* **High Temporal Density:** Generates daily $\text{WSE}$ and associated uncertainty ($\sigma$).

---

## 📦 Repository Structure and Content

The project is organized into dedicated directories for source code, execution, and data management:

| Directory/File | Description |
| :--- | :--- |
| **`model/`** | **Core Source Code.** Contains all essential Python modules defining classes and methods: `River_class.py`, `Station_class.py`, utility modules (`river_utils.py`, `station_utils.py`), data processing modules (`dahiti_data_processing.py`, `gauge_data_processing.py`), and the core logic (`densification_processing.py`, `data_mapping.py`). |
| **`scripts/`** | **Execution.** Contains `run.py`, the main entry point for running the Reach-Reg workflow, defining inputs and calling the densification functions. |
| **`data/`** | **Input Data Storage.** Directory for storing data created by the method, including: intermediate `.pkl` files for river objects, gauge data, vs data, and the `SWORD_v17b_rename/` folder (for RiverSP mapping files). |
| **`results/`** | **Output Results.** Stores final outputs: serialized Reference Station objects (`rs_stations/`) and generated WSE time series and accuracy metrics (`timeseries/`). |
| **`figs_tabs/`** | **Visualizations.** Recommended location for saving charts, plots, and tables generated during analysis or validation. |
| **`.gitignore`** | Ensures all intermediate results, data files (e.g., `.pkl`, large SWORD files), and environment files are excluded from version control. |
| `requirements.txt` | List of external Python dependencies required to run the project. |

---

## 🛠️ How to Use

### 1. Requirements and Setup

The project relies on standard scientific libraries and potentially specialized packages for data access.

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```
### 2. Configuration

All essential user input and method parameters are centralized in two key files:

| File | Purpose | Key Content |
| :--- | :--- | :--- |
| **`scripts/run.py`** | **Workflow Management.** Define the river being processed, the analysis period (TS_START, TS_END), and the output paths. | Defines execution sequence. |
| **`model/data_mapping.py`** | **Core Configuration.** Set river geometry file paths. It is also possible to fine-tune method parameters. | `all_river_data`, `configs` (e.g., `corr_thres`, `amp_thres`). |

#### Data Preparation Instructions

1.  **SWORD Data:** You must download the required river geometry files from the **SWORD database** (Altenau et al., 2021). The path to the downloaded file (specific for a given area) must be defined in `model/data_mapping.py`.
2.  **River Extent:** The start and end reach IDs of the river section to be analyzed can be defined by examining the reaches either visually in a GIS environment or by clicking on the relevant reach on the Dahiti map (`dahiti.dgfi.tum.de/en`).
3.  **RiverSP Data (Optional):** If you intend to use RiverSP data, the necessary mapping files (linking different SWORD versions) must be placed inside the `data/SWORD_v17b_rename/` folder. **[LINK]**

### 3. Execution

The execution is managed through `scripts/run.py`. Users typically call one of the two main orchestration functions from `model/densification_processing.py`:

```python
from model.densification_processing import densify_wl_with_gdata

results_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python_clean/results/'
data_dir = '/Users/michalhalicki/Documents/nauka/projekty/NAWA_BEKKER/Bekker_Python_clean/data/'
rs_dir = results_dir + 'rs_stations/'
ts_dir = results_dir + 'timeseries/'
g_dir = data_dir + 'g_data/'
vs_dir = data_dir + 'vs_data/'
riv_dir = data_dir + 'rivers/'

# Example parameters defined in scripts/run.py
river = 'Elbe'
t1 = '2023-07-10'
t2 = '2025-12-31'

# Run the densification (Recommended: with validation against in-situ data)
densify_wl_at_river_with_gdata(river, t1, t2, rs_dir, g_dir, vs_dir, riv_dir, ts_dir)
# Optional parameter: target_rs_id=None (calculates all RS on a river section. Provide a specific RS id if the densification should be run only for one specific station.)
```
### 4. Output Data Format

The main output time series are found in `results/ts_stations/` as CSV files (`{river}_RS{rs_id}.csv`) for each Reference Station ($\text{RS}$).

| Column | Description | Units |
| :--- | :--- | :--- |
| `wse` | Daily Water Surface Elevation (Reach-Reg estimate) | $\text{m}$ |
| `wse_u` | Total WSE Uncertainty ($\sigma$) | $\text{m}$ |
| `N` | Number of input altimetry measurements aggregated for the day. | count |

---

## 📚 References

* **Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. de M., & Bendezu, L. (2021).** The surface water and ocean topography (SWOT) mission river database (SWORD): A global river network for satellite data products. *Water Resources Research, 57*(7). https://doi.org/10.1029/2021WR030054
* **Halicki, M., Niedzielski, T., Schwatke, C., Scherer, D., & Dettmering, D. (2026).** Daily river water levels from multi-mission altimetry: A reach-based regression method using the unique SWOT data geometry. \[In Review\]
* **Halicki, M., Niedzielski, T., Schwatke, C., Scherer, D., & Dettmering, D. (2025).** Daily Water Surface Elevations on Rivers from Multi-Mission Satellite Altimetry, *Zenodo*, https://doi.org/10.5281/zenodo.17928117
* **Schwatke, C., Dettmering, D., Bosch, W., & Seitz, F. (2015).** Dahiti – an innovative approach for estimating water level time series over inland waters using multi-mission satellite altimetry, *Hydrology and Earth System Sciences, 19*, 4345–4364. https://doi.org/10.5194/hess-19-4345-2015

## 📜 License and Citation

### License (MIT)

This project is licensed under the **MIT License**.

In short: You are free to use, modify, and distribute the code, provided you include the original copyright and license notice.

### Citation

If you utilize the Reach-Reg code or methodology in a publication, project, or presentation, you must cite the corresponding work:

* **Halicki, M., Niedzielski, T., Schwatke, C., Scherer, D., & Dettmering, D. (2026). Daily river water levels from multi-mission altimetry: A reach-based regression method using the unique SWOT data geometry. [In Review]**
