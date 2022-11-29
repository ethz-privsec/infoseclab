#!/bin/bash

#SBATCH -n=4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter.out
#SBATCH --error=jupyter.err
#SBATCH --gpus=1

module load gcc/8.2.0 python_gpu/3.10.4 hdf5/1.10.1 eth_proxy
hostname -i
.venv/bin/jupyter lab --no-browser \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --ip "0.0.0.0" \
    --NotebookApp.disable_check_xsrf=True \
    --allow-root \
    --NotebookApp.port_retries=0