# Introduction

In this tutorial we will demonstrate the process of getting started with the [Unity](https://unity.rc.umass.edu/) compute cluster.
We will show how to setup a conda (python) environment for a project, how to request a job that lanches a jupyther notebook and how to train and evaluate a pytorch model.
The demonstration uses the natural language processing (NLP) task of text summarization as an example, but the workflow can be adapted to any other machine learning model.

**Getting access to Unity**: For this tutorial, we assume that you have access to the Unity cluster. If not, you can follow the procedure mentioned in the [documentation](https://docs.unity.rc.umass.edu/#accessing-unity) to get access.

**Preliminaries**: We will use the [On Demand Portal](https://docs.unity.rc.umass.edu/connecting/ood.html) to get shell access on Unity as well as to lanch the jupyter notebook.
Unity uses [SLURM](https://slurm.schedmd.com/documentation.html) cluster management software. If you are not familiar with clusters, we highly recommend going through the [common terms](https://docs.unity.rc.umass.edu/jargon.html) section of Unity documentation before following this tutorial.

# Using a Jupyter notebook to work with a HuggingFace model for an NLP task

We will use [HuggingFace](https://huggingface.co/), an NLP library that provides models for various NLP tasks.
But before we can use it in our code, we need to install it and other python packages that we require.
This can be done by creating a [conda environment](https://docs.unity.rc.umass.edu/software/conda.html) as shown below. 

## Setting up a conda environment

1. Open a shell on Unity. You can do this by going to the "Shell" tab on the On Demand Portal shown [here](https://docs.unity.rc.umass.edu/connecting/ood.html). This should open a terminal like interface in your browser window.
2. We will now create a conda environment named `hf-summarization` by executing `conda create -n hf-summarization python=3.9.7 pip ipykernel` in the shell that we opened in the previous step. Here we have used python version 3.9 but you can use any suitable python version.
3. Activate the conda environment by executing `conda activate hf-summarization`.
4. Install the required packages using pip: `pip install datasets==2.11.0 transformers==4.28.1 rouge-score nltk` and `pip install torch --index-url https://download.pytorch.org/whl/cu118`. Note that we have installed a pytorch version that uses cuda 11.8; we need to load the same cuda version using the [modules interface](https://docs.unity.rc.umass.edu/software/module-intro.html) while requesting a job with jupyter notebook.
5. Register your conda environment with the global ipykernel: `/modules/apps/ood/jupyterlab/bin/python -m ipykernel install --user --name hf-summarization`.

Now that we have installed the required packages in a conda environment, we are ready to start working on the code!

## Requesting a job with Jupyter notebook

Start by accessing JupyterLab using the [Unity OnDemand interface](https://ood.unity.rc.umass.edu/pun/sys/dashboard/batch_connect/sys/jupyterlab/session_contexts/new).

Click on the JupyterLab interactive app and fill out the following fields:

1. The `Partition` field indicates the type of compute nodes to run your interactive session on. One of the gpu partitions should be selected to run our Jupyter notebook (gpu, gpu-long, uri-gpu or gpu-preempt). For more information on partitions, see [the partition list](https://docs.unity.rc.umass.edu/technical/partitionlist.html).
2. The `Maximum job duration` field defines how long the interactive session with JupyterLab should run for. This field can be left with the default value of one hour (1:00:00) for evaluation but should be increased more than a three hours (3:00:00) if you are going to train a small model, and even more if you are training a bigger model than that one shown in this tutorial. To complete this tutorial, you can set the time limit to 4:00:00 (4 hours).
3. The `Memory (in GB)` field defines the amount of memory (RAM) in gigabytes allocated to your interactive session. To give you an idea of how much memory you may need, 10GB is enough for training as well as evaluation. 
4. The `GPU count` field is the number of GPUs allocated to your interactive session. It should be set to 1 because we will use a single GPU for training and evaluation.
5. The `Modules` field corresponds to a [list of modules](https://docs.unity.rc.umass.edu/software/module-intro.html) to load. The following two modules should be added (separated only by a space) to this field in order to use the GPU for traning and evaluation: `cudnn/cuda11-8.4.1.50 cuda/11.8`
6. The fields `CPU thread count` to 5 (this should be enough for training as well as evaluation).  `Extra arguments for Slurm` can be left blank.

Clicking on "Launch" should start a JupyterLab server on one of the compute nodes.

## Inside JupyterLab:

1. Copy the `summarization-hf.ipynb` notebook on Unity to your work directory. You can use `scp`, `rsync` or the OnDemand portal to copy the file as shown [here](https://docs.unity.rc.umass.edu/managing-files/intro.html). We recommend not using your `/home` directory to store files because it is limited to 50GB only. You can use `/gypsum` (if you have access) or `/work/pi_` directory. Please refer to the [storage](https://docs.unity.rc.umass.edu/technical/storage.html) section of the Unity documentation for more information about directories available for your use.
2. Open the `summarization-hf.ipynb` notebook in JupyterLab.
3. Choose `hf-summarization` as the kernel because it has all our python packages.
4. Now you are ready to execute the code in your notebook. Execute the cells in the notebook following the instructions therein. 
