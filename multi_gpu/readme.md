# Description

This folder contains a script `multi_gpu_finetune.py` that uses `torch.distributed` to finetune GPT2-medium on 1% of the PG19 dataset for 0.1 epoch using 4 GPUs. Since the script was created for demonstration purposes, we only use a very small fraction of the dataset and only train for 0.1 epoch.

If you want to use multiple GPUs in a jupyter notebook, it's not recommended to use `torch.distributed`. Instead, you can look into the Huggingface Accelerate library: https://huggingface.co/docs/accelerate/basic_tutorials/notebook.

# Instructions

1. Create a virtual environment: `python3 -m venv "unity-venv"`

2. Activate the virtual environment: `. ./unity-venv/bin/activate`

3. Make sure you are in the `multi_gpu` folder, then run `pip install -r requirements.txt`

4. Open a new tmux session

5. Schedule a new [srun](https://docs.unity.rc.umass.edu/slurm/srun.html) job with 4 RTX8000 GPUs and 200000 MB of memory: `srun -c 2 -G 4 -p gypsum-rtx8000 --mem 200000 --pty bash`

6. Run the finetuning script, make sure you are in the `multi_gpu` folder: `torchrun --standalone --nnodes=1 --nproc_per_node=4 multi_gpu_finetune.py --output_dir checkpoints --do_train True --do_eval True --evaluation_strategy steps --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_train_epochs 0.1 --save_steps 500 --eval_steps 500`

# Resources

- Using `torchrun`: https://pytorch.org/docs/stable/elastic/run.html
- Huggingface `Trainer` and `TrainingArguments`: https://huggingface.co/docs/transformers/main_classes/trainer
