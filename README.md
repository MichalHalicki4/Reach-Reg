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
| **`model/`** | **Core Source Code.** Contains essential Python modules defining classes, workflow configurations, and processing methods: `Station_class.py`, `config.py` (runtime parameters wrapper), utility modules (`river_utils.py`, `station_utils.py`), data download/parsing modules (`dahiti_data_processing.py`, `gauge_data_processing.py`, `hydrochron_data_processing.py`), and the core densification logic (`densification_processing.py`). || **`scripts/`** | **Execution.** Contains `run.py`, the main entry point for running the Reach-Reg workflow, defining inputs and calling the densification functions. |
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

The Reach-Reg framework is entirely config-driven. Every river section processing setup is defined via a dedicated JSON file (e.g., `config_elbe.json`). This file is parsed at runtime into a nested recursive namespace object (ReachRegConfig), decoupling core code execution from river-specific setups.

Note 1: The `data_provider` key accepts `dahiti` or `hydrocron`. When not provided, the script defaults to the DAHITI database.

Note 2: Provide a specific station ID in `target_rs_id` when aiming to calculate daily WSE for a single given station/reach. Otherwise, set it to `null` (or omit it), and Reach-Reg will sequentially generate daily WSE for all available stations on the river section.

### Structure of a Configuration File

```json
{
    "river_name": "Elbe",
    "target_rs_id": null,
    "river_metadata": {
        "river": "Elbe, River",
        "up_reach": 23285000251,
        "dn_reach": 23281000101,
        "country": "germany",
        "metrical_crs": "4839",
        "vs_with_neight_dams": [
            38476
        ],
        "sword_river_file": "/home/michal.halicki/Desktop/Bekker/SWORD_rivers/SWORD_v17b_shp/EU/eu_sword_reaches_hb23_v17b.shp",
        "river_tributary_reaches": [],
        "gauge_dist_threshold": 5
    },
    "model_configs": {
        "amp_thres": 1,
        "corr_thres": 0.75,
        "itpd_method": "akima"
    },
    "temporal_range": {
        "t1": "2023-01-01 00:00",
        "t2": "2026-01-01 00:00"
    },
    "validate_with_gauge": true,
    "data_provider": "dahiti"
}
```

#### Data Preparation Instructions

1.  **SWORD Data:** You must download the required river geometry files from the **SWORD database** (Altenau et al., 2021). The path to the downloaded file (specific for a given area) must be defined in `model/data_mapping.py`.
2.  **River Extent:** The start and end reach IDs of the river section to be analyzed can be defined by examining the reaches either visually in a GIS environment or by clicking on the relevant reach on the Dahiti map (`dahiti.dgfi.tum.de/en`).
3.  **RiverSP Data (Optional):** If you intend to use RiverSP data, the necessary mapping files (linking different SWORD versions) must be placed inside the `data/SWORD_v17b_rename/` folder. It can be downloaded from the SWORD Explorer website (https://www.swordexplorer.com/, Download --> SWORD v16 to SWORD v17b Translation).

### 3. Execution

Execution is triggered by feeding a specific JSON configuration directly into `run.py`. The workflow automatically detects whether validation against in-situ gauges is requested (`validate_with_gauge`) and instantiates directory tree paths dynamically.

```bash
python run.py config_elbe.json
```

For mass production or automated pipeline runs, trigger the provided shell script orchestrator:

```bash
chmod +x run_multiple_rivers.sh
./run_multiple_rivers.sh
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
* **Halicki, M., Niedzielski, T., Schwatke, C., Scherer, D., & Dettmering, D. (2026).** Daily river water levels from multi-mission altimetry: A reach-based regression method using the unique SWOT data geometry. *Journal of Hydrology 673*, 135367. https://doi.org/10.1016/j.jhydrol.2026.135367
* **Halicki, M., Niedzielski, T., Schwatke, C., Scherer, D., & Dettmering, D. (2025).** Daily Water Surface Elevations on Rivers from Multi-Mission Satellite Altimetry, *Zenodo*, https://doi.org/10.5281/zenodo.17928117
* **Schwatke, C., Dettmering, D., Bosch, W., & Seitz, F. (2015).** Dahiti – an innovative approach for estimating water level time series over inland waters using multi-mission satellite altimetry, *Hydrology and Earth System Sciences, 19*, 4345–4364. https://doi.org/10.5194/hess-19-4345-2015

## 📜 License and Citation

### License (MIT)

This project is licensed under the **MIT License**.

In short: You are free to use, modify, and distribute the code, provided you include the original copyright and license notice.

### Citation

If you utilize the Reach-Reg code or methodology in a publication, project, or presentation, you must cite the corresponding work:

* Halicki, M., Niedzielski, T., Schwatke, C., Scherer, D., & Dettmering, D. (2026). Daily river water levels from multi-mission altimetry: A reach-based regression method using the unique SWOT data geometry. *Journal of Hydrology 673*, 135367. https://doi.org/10.1016/j.jhydrol.2026.135367
