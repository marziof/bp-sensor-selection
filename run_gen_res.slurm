#!/bin/bash
#SBATCH --job-name=gen_results
#SBATCH --output=gen_results.out
#SBATCH --error=gen_results.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=academic

python gen_results.py