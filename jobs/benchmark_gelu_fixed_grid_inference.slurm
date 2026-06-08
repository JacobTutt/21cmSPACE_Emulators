#!/bin/bash
#SBATCH -J gelu_infer_bench
#SBATCH --account=acedo-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -o logs/gelu_infer_bench_%j.out
#SBATCH -e logs/gelu_infer_bench_%j.err

set -euo pipefail

BASE=/rds/user/jlt67/hpc-work/PhD/NenuFAR
REPO="${BASE}/21cmSPACE_Emulators"
ENV="${BASE}/conda-env"

cd "${REPO}"
mkdir -p logs

module load miniconda/3
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV}"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python jobs/benchmark_gelu_fixed_grid_inference.py
