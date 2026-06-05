#!/bin/bash
#SBATCH --job-name=gen_results
#SBATCH --output=gen_results.out
#SBATCH --error=gen_results.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=academic

export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

cd $SLURM_SUBMIT_DIR

echo "Running in: $(pwd)"
python scripts/gen_res_new.py