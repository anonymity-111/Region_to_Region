unset WORLD_SIZE
python train_control.py \
        --ckpt_name diff-base  \
        --save_name Diff-test \
        --max_epochs 5 \
        --config cldm_v15_cn \
        --dataset ihar \
