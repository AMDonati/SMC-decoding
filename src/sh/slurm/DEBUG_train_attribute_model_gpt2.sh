#!/bin/bash
#SBATCH --job-name=DEBUG-gpt2
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/sst/DEBUG_gpt2%j.out
#SBATCH --error=slurm_out/sst/DEBUG_gpt2%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-dec

export PYTHONPATH=src:${PYTHONPATH}

srun python -u src/attribute_models/train_attribute_model.py -out_path "output/DEBUG" -model "gpt2" -label 1 -p_drop 0.0 -ep 1 -bs 16 -num_workers 4 -init_weight 0 -lr 0.00005
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/DEBUG" -model "gpt2" -label 1 -p_drop 0.0 -ep 1 -bs 16 -num_workers 4 -init_weight 1 -lr 0.00005
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/DEBUG" -model "gpt2" -label 1 -p_drop 0.0 -ep 1 -bs 16 -num_workers 4 -init_weight 1 -KL_coeff 0.1 -lr 0.00005




