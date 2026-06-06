#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5|dlg6

#SBATCH --job-name="c100_h"
#SBATCH --account=punim1623
#SBATCH --time=6-10:00:00

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
###### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --cpus-per-task=1
#SBATCH --mem=512G

export WORLD_SIZE=4   ####FIXME: update world size: nodes x ntasks-per-node
export MASTER_PORT=12340

echo ">>> NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

srun python -u main_train.py \
    --method byol \
    --trigger_type htba \
    --arch vit_b_16 \
    --image_size 224 \
    --pretrain_batch_size 32 \
    --lr 1.5e-4 \
    --wd 0.1 \
    --temp 0.1 \
    --knn_eval_freq 100 \
    --model_save_freq 100 \
    --warmup_epoch 40 \
    \
    --magnitude_train 300.0 \
    --magnitude_val 300.0 \
    --dataset imagenet100 \
    --target_class 26 \
    --use_ibc \
    --find_channels_from_n_poison_samples 4 \
    --find_channels_from_n_clean_samples 0 \
    
    \
    #--pretrained_ssl_model "./Experiments/cifar100_htba_byol_ref_set_sd42/last.pth.tar" \
    #--pretrained_linear_model "./Experiments/cifar100_htba_byol_ref_set_sd42/linear.pth.tar" \

##Log this job's resource usage stats###
my-job-stats -a -n -s