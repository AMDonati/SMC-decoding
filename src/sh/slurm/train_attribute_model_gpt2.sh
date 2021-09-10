#!/bin/bash
#SBATCH --job-name=sst-gpt2
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/sst/sst_gpt2%j.out
#SBATCH --error=slurm_out/sst/sst_gpt2%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-dec

export PYTHONPATH=src:${PYTHONPATH}

#srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "gpt2" -label 1 -p_drop 0.0 -ep 50 -bs 32 -num_workers 4
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "gpt2" -label 1 -p_drop 0.0 -ep 50 -bs 32 -num_workers 4 -lr 0.00005 -grad_clip 1
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "gpt2" -label 1 -p_drop 0.0 -ep 50 -bs 32 -num_workers 4 -lr 0.00005 -grad_clip 5
#srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "gpt2" -label 1 -p_drop 0.0 -ep 50 -bs 16 -num_workers 4
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "gpt2" -label 1 -p_drop 0.0 -ep 50 -bs 16 -num_workers 4 -lr 0.00005 -grad_clip 1
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "gpt2" -label 1 -p_drop 0.0 -ep 50 -bs 16 -num_workers 4 -lr 0.00005 -grad_clip 5
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "gpt2" -label 1 -p_drop 0.0 -ep 50 -bs 64 -num_workers 4 -lr 0.00005
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "gpt2" -label 1 -p_drop 0.0 -ep 50 -bs 64 -num_workers 4 -lr 0.00005 -grad_clip 1
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "gpt2" -label 1 -p_drop 0.0 -ep 50 -bs 64 -num_workers 4 -lr 0.00005 -grad_clip 5
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "gpt2" -label 1 -p_drop 0.0 -ep 50 -bs 128 -num_workers 4 -lr 0.00005
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "gpt2" -label 1 -p_drop 0.0 -ep 50 -bs 128 -num_workers 4 -lr 0.00005 -grad_clip 1
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "gpt2" -label 1 -p_drop 0.0 -ep 50 -bs 128 -num_workers 4 -lr 0.00005 -grad_clip 5
