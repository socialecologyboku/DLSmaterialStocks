# DLS Material Stocks

Analysis of material requirements for decent living standards globally.

This is the code repository for the paper: **"Small increases in society's material stocks to achieve decent living standards globally"** by Streeck et al.

## Installation

### Setup Instructions

1. **Create a new virtual environment, e.g. via conda:**

   ```bash
   conda create -n dls-materials python=3.11
   conda activate dls-materials
   ```

   This new environment needs to have Python / pip installed.
2. **Install the package in development mode:**

   ```bash
   pip install -e .
   ```

   This will install all necessary requirements to run the code. The -e flag installs in development mode, which means you can alter the code and it will reload automatically.
3. **Verify installation:**

   ```bash
   conda list dls-material-stocks
   ```

## Usage

The repository contains all the input files needed to run different DLS scenarios. The scenario data is created via scripts/create_scenario_data.py. This will create pickles of the input data needed to compute the scenarios. Once created, you can use scripts/run_scenarios.py to examine different combinations of practices and thresholds. Running a scenario will save the entire outputs in an .xlsx file and create all figures used in the article for that specific combination. By default, displaying the figures is disabled - activate a matplotlib backend to show them.

In the scripts all scenarios are active, which results in a fairly long runtime. Adapt this to your needs if you are interested in only parts of the output.

### **Project Structure**

```
src/dls_material_stocks/           # main source code package
├── analysis/                      # core analysis modules
│   ├── analysis_flexible_v1.py    # main analysis script with configurable conv_gap_mode
│   ├── analysis_flexible_v1_convGaps.py  # specialized convGaps analysis variant
│   ├── DLS_functions_v3.py        # analysis utility functions
│   └── DLS_materials_bottomUp_*.py # bottom-up material calculations
├── harmonization/                 # data harmonization utilities
│   └── harmonization.py           # material and stock harmonization
├── load/                          # data loading modules
│   └── DLS_load_data.py           # data loading functions
├── plots/                         # visualization modules
│   ├── DLS_plots.py               # plotting functions
│   └── aggregate_thresholds_results.py # threshold aggregation plots
└── scenarios/                     # scenario configuration
    └── scenario_config.py         # scenario definitions

scripts/                           # main execution scripts
├── create_scenario_data.py        # generate scenario input data (pickles)
└── run_scenarios.py               # run analysis scenarios

input/                             # input data files
├── 2021_Kikstra/                  # DLS threshold and indicator data
├── 2023_Velez/                    # material intensity data
├── MISO2/                         # MISO2 material flow data
├── country_correspondence.xlsx     # country mapping
└── [various scenario data]/       # practice-specific input files

output/                            # analysis results
├── scenario_data/                 # generated pickle files
└── results_*.xlsx                 # analysis output files
```

## License

See LICENSE.txt for details.

## Citation

If you use this code, please cite:

```
Streeck et al. "Small increases in society's material stocks to achieve decent living standards globally"
```
