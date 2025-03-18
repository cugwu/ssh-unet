CUDA_LAUNCH_BLOCKING=1 python -u main.py --save_checkpoint --logdir "./logdir" --data_dir "./btcv" --json_list "btcv_dataset.json" \
                 --roi_x 96 --roi_y 96 --roi_z 96 --in_channels 1 --out_channels 14 --nfolds 1 --kernels 133 133 133 133 133 --gate_type "tsm" \
                 --gate_pos 0 1 2 --gate_bottleneck --gate_dec --batch_size 2 --max_epochs 5000 --val_every 100 --patience 1000 --optim_name "adamw" --optim_lr 4e-4 \
                 --cache_num 30 --pos 1 --neg 1 --a_min -175.0 --a_max 250.0 --b_min 0.0 --b_max 1.0 --RandFlipd_prob 0.1 \
                 --RandRotate90d_prob 0.1 --RandScaleIntensityd_prob 0.5 --RandShiftIntensityd_prob 0.5

