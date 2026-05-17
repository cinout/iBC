#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5|dlg6

#SBATCH --job-name="img_f"
#SBATCH --account=punim1623
#SBATCH --time=0-00:30:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=120G

# export WORLD_SIZE=2   ### update world size: nodes x ntasks-per-node
# export MASTER_PORT=28400
# echo ">>> NODELIST="${SLURM_NODELIST}
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

python -u main_train.py \
    --method byol \
    \
    --image_size 64 \
    \
    --magnitude_train 300.0 \
    --magnitude_val 300.0 \
    --dataset imagenet100 \
    --target_class 26 \
    \
    \
    --use_ibc \
    --find_channels_from_n_poison_samples 3 \
    --find_channels_from_n_clean_samples 5 \
    \
    --pretrained_ssl_model "./Experiments/imagenet100_ftrojan_byol_ref_set_sd42/last.pth.tar" \
    --pretrained_linear_model "./Experiments/imagenet100_ftrojan_byol_ref_set_sd42/linear.pth.tar" \
    

##Log this job's resource usage stats###
my-job-stats -a -n -s