#!/bin/bash

CLASS_ARRAY=(92 95 146 69 138 116 148 96 122 61 149 142 102 124 94 53 107 105 109 132 85 42 54 60)


# Number of parallel jobs (adjust based on how many GPUs or jobs your node can handle)
MAX_JOBS=1

# Track background job count
job_count=0

for ARRAY_INDEX in "${!CLASS_ARRAY[@]}"; do
    CLASS_ID=${CLASS_ARRAY[$ARRAY_INDEX]}
    WORK_DIR=/work3/s203557/outputs_box_test/test_results_class_${CLASS_ID}

    echo "Launching class $CLASS_ID..."

    CUDA_VISIBLE_DEVICE=2 python segmentation/test.py segmentation/configs/vpd_config.py \
        /work3/s203557/experiments/control_net_vpd/iter_2000.pth \
        --class-filter $CLASS_ID \
        --work-dir $WORK_DIR \
        --eval mIoU \


    ((job_count++))

    # Limit number of concurrent jobs
    if (( job_count >= MAX_JOBS )); then
        wait -n  # Wait for any job to finish
        ((job_count--))
    fi
done

# Wait for remaining jobs
wait
echo "All evaluations complete."