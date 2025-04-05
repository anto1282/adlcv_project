#!/bin/bash
#BSUB -q gpuv100
#BSUB -J single_inference_VPD_short
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:15
#BSUB -R "rusage[mem=4GB]"
#BSUB -u s203557@dtu.dk
#BSUB -B
#BSUB -env "LSB_JOB_REPORT_MAIL=N"
#BSUB -N
#BSUB -o %J.out
#BSUB -e %J.err

# Load necessary modules
module load python3/3.11.4
module load cuda/11.3

# Activate virtual environment
source /dtu/blackhole/0e/154958/miniconda3/bin/activate ldm


python single_inference.py 