#!/usr/bin/env bash
# Train an AR transformer on discrete VideoRAE tokens.
# Edit VAE_CKPT / OUT_PATH before running.

VAE_CKPT="checkpoints/videorae_discrete.pth"
OUT_PATH="save/ar_vjepa"

python3 \
    train.py --cfg cfgs/vfm_ar_repa.yaml \
    --manualSeed 66667 --tag default \
    --csv_file ucf101_train.csv --out_path ${OUT_PATH} \
    --name vfm_ar_repa -b 64 -j 64 \
    --frame_num 16 --input_size 256 \
    --opts \
    test_dataset.csv_paths.ucf101_val ucf101_val.csv \
    model.name llama-abs-XXL \
    vae.checkpoint ${VAE_CKPT} \
    vae.version sd \
    ar.num_samples 32 \
    optimizer.name adamw \
    optimizer.args.weight_decay 0.05 \
    optimizer.warmup_epoch 4 \
    optimizer.args.lr 0.0006 \
    use_amp true \
    compile true \
    vis_epoch 5 eval_epoch 5 max_epoch 3000 latest_interval 5
