#!/bin/bash
#SBATCH --job-name=scar-ep-pinn-case1
#SBATCH --partition=hgx2q,a100q,h200q
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00-04:00:00
#SBATCH --output=slurm_output/slurm%j.out
#SBATCH --error=slurm_output/slurm%j.err

ulimit -v unlimited

module purge

current_path=$(pwd)

ARCH=$(uname -m)
echo "Architecture: $ARCH"

export WORKON_HOME="$HOME/.virtualenvs/$ARCH"
pipenv sync

pipenv run python train_pinn.py --config config_multicase.yaml --case case_1