## README

This project makes use of the bpepi library from https://github.com/SPOC-group/bpepi.
Some code from https://github.com/IdePHICS/BPEpI-results/tree/main is also reused.


## 📋 Overview

This project provides implementations of sensor selection algorithms for inference of epidemic spreading processes using BP.

**Repository:** [marziof/deep-rl-project](https://github.com/marziof/bp-sensor-selection)

##  Installation

### Prerequisites
- check out requirements.txt
- you will need the bpepi package from https://github.com/SPOC-group/bpepi

### Setup

```bash
# Clone the repository
git clone https://github.com/marziof/bp-sensor-selection.git
cd bp-sensor-selection

# Create virtual environment
python -m venv sensor_selection_env # or with conda
source sensor_selection_env/bin/activate  # On Windows: rl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Configure an experiment

Edit or create a configuration file in `configs/`; choose:
- parameters
- methods/metrics
- save directory

### 2. Run an experiment

```bash
python scripts/gen_results.py 
```
or from cluster with run_gen_results.slurm

Results are saved to a chosen directory (here, 'results')

### 3. Plot results

```bash
python srcipts/plot_results.py # for overlap curves
python srcipts/plot_sensors.py # for selected sensor properties
```

Plots are saved in 'results/plots'

  
## Project structure
'''
.
├── configs
│   └── default.py                                 # set configurations for simulation (algorithm to use, parameters, graph, etc)
├── results
│   ├── dfs                                        # metric results
│   ├── past_tests
│   ├── plots
│   └── sensor_stats                               # properties of selected nodes for tracking
├── scripts
│   ├── gen_results.py                             # Main file to run simulations with configs 
│   ├── plot_pinf_profile.py
│   ├── plot_results.py
│   ├── plot_sensors.py
│   └── run_gen_results.slurm                      # To run gen_results.py  
├── src
│   ├── algorithms
│   │   ├── non_oracle_selection.py
│   │   ├── optimal_subset_selection.py
│   │   ├── sequential_sensor_selection.py
│   │   └── static_selection.py
│   ├── Analysis                                   # From https://github.com/IdePHICS/BPEpI-results/tree/main
│   │   ├── gen.py
│   │   ├── measures.py
│   │   ├── sim_oc_dSIR.py
│   │   └── XZtoDF.py
│   ├── experiments                                # function to loop over algorithms and parameters, called in gen_results
│   │   └── full_sweep_new.py
│   ├── helpers
│   │   ├── algo_helpers.py
│   │   ├── pipeline_helpers.py
│   │   ├── plot_helpers.py
│   │   ├── plot_sensor_stats.py
│   │   └── sim_graph.py
│   └── utils
│       ├── metrics.py                            
│       └── sensor_logger.py
└── test_nb.ipynb

15 directories, 209 files
(base) marzioformica@Marzios-MacBook-Pro sensorSelection % 
'''
