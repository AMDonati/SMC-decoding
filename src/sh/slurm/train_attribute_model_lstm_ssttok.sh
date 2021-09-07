#!/bin/bash
#SBATCH --job-name=sst-lstm
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/sst/sst_lstm%j.out
#SBATCH --error=slurm_out/sst/sst_lstm%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-dec

export PYTHONPATH=src:${PYTHONPATH}

#srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "lstm" -tokenizer "sst" -label 1 -emb_size 64 -hidden_size 128 -p_drop 0.0 -ep 100 -bs 128 -num_workers 4
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "lstm" -tokenizer "sst" -label 1 -emb_size 64 -hidden_size 128 -p_drop 0.0 -ep 100 -bs 128 -num_workers 4 -label_vocab 1
#srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "lstm" -tokenizer "sst" -label 1 -emb_size 64 -hidden_size 128 -p_drop 0.0 -ep 100 -bs 128 -num_workers 4 -min_count 1
srun python -u src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "lstm" -tokenizer "sst" -label 1 -emb_size 64 -hidden_size 128 -p_drop 0.0 -ep 100 -bs 128 -num_workers 4 -label_vocab 1

