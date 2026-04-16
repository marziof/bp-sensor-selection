#!/bin/bash
#SBATCH --job-name=gen_results
#SBATCH --output=gen_results.out
#SBATCH --error=gen_results.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

module load python   # only if your cluster uses modules
python gen_results.py