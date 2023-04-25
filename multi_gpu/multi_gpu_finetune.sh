#!/bin/bash
#SBATCH -c 2  # Number of CPU cores
#SBATCH --mem=100000  # Requested Memory
#SBATCH -p gypsum-rtx8000  # Partition
#SBATCH -G 2  # Number of GPUs
#SBATCH -o slurm-%j.out  # Specify where to save terminal output, where %j = job ID

# Load modules
module load pytorch

torchrun --standalone --nnodes=1 --nproc_per_node=2 multi_gpu_finetune.py --output_dir checkpoints --do_train True --do_eval True --evaluation_strategy steps --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_train_epochs 0.1 --save_steps 500 --eval_steps 500