# Description

This folder contains a script `multi_gpu_finetune.py` that uses `torch.distributed` to finetune GPT2-medium on 1% of the PG19 dataset for 0.1 epoch using 4 GPUs. Since the script was created for demonstration purposes, we only use a very small fraction of the dataset and only train for 0.1 epoch. The following code is intended to be run from the terminal instead of the on-demand interface.

If you want to use multiple GPUs in a jupyter notebook, it's not recommended to use `torch.distributed`. Instead, you can look into the Huggingface Accelerate library: https://huggingface.co/docs/accelerate/basic_tutorials/notebook.

# Preparation

1. Run `cd multi_gpu` to make sure you are in the `multi_gpu` folder.

2. Create a new virtual environment: `python3 -m venv "unity-venv"`.

3. Activate the virtual environment: `. ./unity-venv/bin/activate`.

4. Run `pip install -r requirements.txt`, this will take a few minutes.

# Scheduling a SRUN job

SRUN is a blocking command, which means that it will not let you execute other commands until this command has finished. Outputs will be directly printed in your terminal. It's recommended to use SRUN during the debugging stage. See [here](https://docs.unity.rc.umass.edu/slurm/srun.html) for the official Unity documentation on SRUN jobs.

1. Open a new tmux session: `tmux`.

2. Schedule a new SRUN job with 2 RTX8000 GPUs and 100000 MB of memory: `srun -c 2 -G 2 -p gypsum-rtx8000 --mem 100000 --pty bash`.

3. You will need to activate the virtual environment again: `. ./unity-venv/bin/activate`.

4. Run the finetuning Python script: `torchrun --standalone --nnodes=1 --nproc_per_node=2 multi_gpu_finetune.py --output_dir checkpoints --do_train True --do_eval True --evaluation_strategy steps --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_train_epochs 0.1 --save_steps 500 --eval_steps 500`.

    - `standalone`: indicates that we are running one instance (job)
    - `nnodes`: the number of GPU node you are using
    - `nproc_per_node`: number of processes per GPU node, this should be equal to the value of the `-G` flag in the `srun` command, otherwise you will get an `CUDA: invalid device ordinal` error

5. After the job is successfully launched, it will first run tokenization on the dataset, and you should see a progress bar that says "Running tokenizer on dataset". The progress bar will likely remain at 0%, but the tokenization is taking place. It will take a while before it finishes processing the data. After that, finetuning will start, and you should observe a new progress bar. Feel free to kill the running script with `ctrl + c` after you see the tokenization progress bar, there is no need to finish the training process. 

5. To terminate a SRUN job, you can simply kill the tmux session by running `tmux kill-session`.

# Scheduling a SBATCH job

SBATCH is a non-blocking command, which means that you will not be blocked from running other commands. If the resources you request for the job are not immediately available, the job will be added to a queue. In the meantime, you can run other commands in your terminal. Outputs will be written to a specified output file. It's recommended to use SBATCH when you have finished debugging your code and are ready to formally launch training or finetuning. See [here](https://docs.unity.rc.umass.edu/slurm/sbatch.html) for the official Unity documentation on SBATCH jobs.

1. Run the SBATCH bash script: `sbatch multi_gpu_finetune.sh`.

2. The terminal should print `Submitted batch job JOBID`, where `JOBID` is the ID of your submitteds job. You can check the status of this job using `sacct -j JOBID`. You can also view all your queued and running jobs using `squeue --me`.

3. A file named `slurm-JOBID.out` should have been created (we specified this output file with the `-o` flag in the bash script), all terminal output that you would observe if running a SRUN job will instead be continuously written to this file.

4. To terminate a SBATCH job, run `scancel JOBID`.

# Resources

- Using `torchrun`: https://pytorch.org/docs/stable/elastic/run.html
- Huggingface `Trainer` and `TrainingArguments`: https://huggingface.co/docs/transformers/main_classes/trainer
