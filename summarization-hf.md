# Introduction to HuggingFace

TODO

# Using a Jupyter notebook to work with a HuggingFace model for an NLP task

## Setting up a conda environment

1. Open a shell on unity. You can use the On Demand Portal for this as shown [here](https://docs.unity.rc.umass.edu/connecting/ood.html).
2. Then create a conda environment named `hf-summarization` using `conda create -n hf-summarization python=3.9.7 pip ipykernel`. 
3. Activate the conda environment `conda activate hf-summarization`.
4. Install the required packages using pip: `pip install datasets transformers rouge-score nltk` and `pip install torch --index-url https://download.pytorch.org/whl/cu118`. Note that we have installed a pytorch version that uses cuda 11.8. We need to load the same cuda version while requesting a job with jupyter notebook.
5. Register your conda environment with the global ipykernel: `/modules/apps/ood/jupyterlab/bin/python -m ipykernel install --user --name hf-summarization`.
6. Make sure that you load the right cuda version `module load cuda/11.8` before starting jupyter. This can be done using the job dashboard when you request a job for your jupyter notebook.


## Requesting a job with Jupyter notebook

Start by accessing JupyterLab using the [Unity OnDemand interface](https://ood.unity.rc.umass.edu/pun/sys/dashboard/batch_connect/sys/jupyterlab/session_contexts/new).

Click on the JupyterLab interactive app and fill out the following fields:

1. The `Partition` field indicates the type of compute nodes to run your interactive session on. One of the gpu partitions should be selected to run ColabFold Jupyter notebook (gpu, gpu-long, uri-gpu or gpu-preempt). For more information on partitions, see [the partition list](https://docs.unity.rc.umass.edu/technical/partitionlist.html).
2. The `Maximum job duration` field defines how long the interactive session with JupyterLab should run for. This field can be left with the default value of one hour (1:00:00) for evaluation but should be increased more than a three hours (3:00:00) if you are going to train a small model, and even more if you are training a bigger model than that one shown in this tutorial. 
3. The `Memory (in GB)` field defines the amount of memory (RAM) in gigabytes allocated to your interactive session. To give you an idea of how much memory you may need, 10GB is enough for training as well as evaluation. 
4. The `GPU count` field is the number of GPUs allocated to your interactive session. It should be set to 1 because this tutorial is about using a single GPU for training and evaluation.
5. The `Modules` field corresponds to a list of modules to load. The two following modules should be added (separated only by a space) to this field in order to use the GPU: `cudnn/cuda11-8.4.1.50 cuda/11.8`
6. The fields `CPU thread count` to 5 (this should be enough for training as well as evaluation).  `Extra arguments for Slurm` can be left blank.

## Inside JupyterLab:

1. Copy the `summarization-hf.ipynb` notebook on Unity to your work directory.
2. Open the `summarization-hf.ipynb` notebook.
3. Choose `hf-summarization` for the kernel.
4. Execute the cells in the notebook following the instructions therein. 
