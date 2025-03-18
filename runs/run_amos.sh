CUDA_LAUNCH_BLOCKING=1 python -u main.py --save_checkpoint --logdir "./logdir" --data_dir "./amos22" --json_list "task01_dataset.json" \
                --roi_x 96 --roi_y 96 --roi_z 96 --out_channels 16 --nfolds 1 --kernels 133 133 133 133 133 --gate_type "tsm" \
                --gate_pos 0 1 2 --gate_bottleneck --gate_dec --max_epochs 1000 --val_every 100 --optim_name "sgd" --optim_lr 0.01 --batch_size 1 \
                --cache_num 100 --momentum 0.99 --a_min -991.0 --a_max 362.0 --b_min 0.0 --b_max 1.0 \
                --RandFlipd_prob 0.2 --RandRotate90d_prob 0.2 --RandScaleIntensityd_prob 0.5 --RandShiftIntensityd_prob 0.5

