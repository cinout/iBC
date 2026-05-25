#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5|dlg6

#SBATCH --job-name="c100_f"
#SBATCH --account=punim1623
#SBATCH --time=8-00:00:00

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
###### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --cpus-per-task=1
#SBATCH --mem=80G

export WORLD_SIZE=4   ####TODO: update world size: nodes x ntasks-per-node
export MASTER_PORT=12340

echo ">>> NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

srun python -u main_train.py \
    --method simclr \
    --trigger_type ftrojan \
    --arch vit_b_16 \
    --pretrain_batch_size 32 \
    --lr 3e-4 \
    \
    --magnitude_train 50.0 \
    --magnitude_val 100.0 \
    --dataset imagenet100 \
    --target_class 0 \
    
    \
    #--pretrained_ssl_model "./Experiments/cifar100_ftrojan_simclr_ref_set_sd42/last.pth.tar" \
    #--pretrained_linear_model "./Experiments/cifar100_ftrojan_simclr_ref_set_sd42/linear.pth.tar" \


##Log this job's resource usage stats###
my-job-stats -a -n -s