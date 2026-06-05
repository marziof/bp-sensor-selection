# BP Sensor Selection

This project implements sensor selection algorithms for inference of epidemic spreading processes using Belief Propagation (BP).

## Dependencies

This project depends on the **bpepi** library:

https://github.com/SPOC-group/bpepi

Some analysis code is adapted from:

https://github.com/IdePHICS/BPEpI-results/tree/main

## Overview

The repository provides implementations and evaluation pipelines for static, sequential, and optimization-based sensor selection methods for epidemic source inference and state estimation.

## Installation

### Prerequisites

- Python 3.11+
- `bpepi` installed from source
- Packages listed in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone https://github.com/marziof/bp-sensor-selection.git
cd bp-sensor-selection

# Create virtual environment (or conda env)
python -m venv sensor_selection_env
source sensor_selection_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Configure an experiment

Edit or create a configuration file in `configs/`.

Typical configuration options include:

- Graph type and parameters
- Epidemic model parameters
- Sensor selection algorithm
- Evaluation metrics
- Output directory

### 2. Run experiments

```bash
python scripts/gen_results.py
```

or on a cluster:

```bash
sbatch scripts/run_gen_results.slurm
```

Results are saved in the configured output directory (typically `results/`).

### 3. Generate plots

```bash
python scripts/plot_results.py
python scripts/plot_sensors.py
```

Generated figures are stored in:

```text
results/plots/
```

## Project Structure

```text
.
├── configs
│   └── default.py
├── results
│   ├── dfs
│   ├── past_tests
│   ├── plots
│   └── sensor_stats
├── scripts
│   ├── gen_results.py
│   ├── plot_pinf_profile.py
│   ├── plot_results.py
│   ├── plot_sensors.py
│   └── run_gen_results.slurm
├── src
│   ├── algorithms
│   │   ├── non_oracle_selection.py
│   │   ├── optimal_subset_selection.py
│   │   ├── sequential_sensor_selection.py
│   │   └── static_selection.py
│   ├── Analysis
│   ├── experiments
│   ├── helpers
│   └── utils
└── test_nb.ipynb
```
