#!/bin/bash
#BSUB -q gpuv100
#BSUB -J dist_train_vpd
#BSUB -n 16
#BSUB -gpu "num=4:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "select[sxm2]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "select[gpu80gb]s"

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

# Run distributed training using your script
bash segmentation/tools/dist_train.sh segmentation/configs/vpd_config.py 2 \
  --load-from /work3/s203557/checkpoints/vpd.chkpt
