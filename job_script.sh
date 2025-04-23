#!/bin/bash
#BSUB -q gpua100
#BSUB -J train_vpd_controlNet
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=16GB]"
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


python segmentation/train.py segmentation/configs/vpd_config.py --load-from /work3/s203557/checkpoints/vpd.chkpt 