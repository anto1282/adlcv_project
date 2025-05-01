#!/bin/bash
#BSUB -q gpuv100
##BSUB -q gpua100
#BSUB -J test_diffSAM_dot[1-24]
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:05
#BSUB -R "rusage[mem=16GB]"
#BSUB -u s203557@dtu.dk
#BSUB -B
#BSUB -env "LSB_JOB_REPORT_MAIL=N"
#BSUB -N
#BSUB -o output/%J_%I.out
#BSUB -e output/%J_%I.err

# Load necessary modules
CLASS_ARRAY=(92 95 146 69 138 116 148 96 122 61 149 142 102 124 94 53 107 105 109 132 85 42 54 60)
module load python3/3.11.4
module load cuda/11.3

INDEX=$LSB_JOBINDEX
ARRAY_INDEX=$((INDEX - 1))

# Activate virtual environment
source /dtu/blackhole/0e/154958/miniconda3/bin/activate ldm

WORK_DIR=/work3/s203557/outputs_dot/test_results_class_${CLASS_ARRAY[$ARRAY_INDEX]}

# Correct call to test.py
python segmentation/test.py segmentation/configs/vpd_config.py /work3/s203557/experiments/control_net_vpd/iter_12000.pth --class-filter ${CLASS_ARRAY[$ARRAY_INDEX]} --work-dir ${WORK_DIR} --eval mIoU --input-type dot
