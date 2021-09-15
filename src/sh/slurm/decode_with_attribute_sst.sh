#!/bin/bash
#SBATCH --job-name=dec-sst
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/sst/dec-sst-%j.out
#SBATCH --error=slurm_out/sst/dec-sst-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-dec

export PYTHONPATH=src:${PYTHONPATH}

MODEL_PATH = "output/models/gpt2ft_sst1/1/model.pt"
OUT_PATH = "output/sst_decoding"
NOISE_FUNCTION = "constant"

srun python -u src/scripts/decode_with_attribute.py -model_path $MODEL_PATH -out_path $OUT_PATH -std 0.05 -noise_function $NOISE_FUNCTION
srun python -u src/scripts/decode_with_attribute.py -model_path $MODEL_PATH -out_path $OUT_PATH -std 0.1 -noise_function $NOISE_FUNCTION
srun python -u src/scripts/decode_with_attribute.py -model_path $MODEL_PATH -out_path $OUT_PATH -std 0.25 -noise_function $NOISE_FUNCTION
srun python -u src/scripts/decode_with_attribute.py -model_path $MODEL_PATH -out_path $OUT_PATH -std 0.5 -noise_function $NOISE_FUNCTION
srun python -u src/scripts/decode_with_attribute.py -model_path $MODEL_PATH -out_path $OUT_PATH -std 1 -noise_function $NOISE_FUNCTION
srun python -u src/scripts/decode_with_attribute.py -model_path $MODEL_PATH -out_path $OUT_PATH -std 5 -noise_function $NOISE_FUNCTION

NOISE_FUNCTION = "decreasing"

srun python -u src/scripts/decode_with_attribute.py -model_path $MODEL_PATH -out_path $OUT_PATH -std 0.05 -noise_function $NOISE_FUNCTION
srun python -u src/scripts/decode_with_attribute.py -model_path $MODEL_PATH -out_path $OUT_PATH -std 0.1 -noise_function $NOISE_FUNCTION
srun python -u src/scripts/decode_with_attribute.py -model_path $MODEL_PATH -out_path $OUT_PATH -std 0.25 -noise_function $NOISE_FUNCTION
srun python -u src/scripts/decode_with_attribute.py -model_path $MODEL_PATH -out_path $OUT_PATH -std 0.5 -noise_function $NOISE_FUNCTION
srun python -u src/scripts/decode_with_attribute.py -model_path $MODEL_PATH -out_path $OUT_PATH -std 1 -noise_function $NOISE_FUNCTION
srun python -u src/scripts/decode_with_attribute.py -model_path $MODEL_PATH -out_path $OUT_PATH -std 5 -noise_function $NOISE_FUNCTION