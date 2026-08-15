# Train continuous VideoRAE (V-JEPA 2) on an 8-GPU machine with DDP.
# Pass model.args.vjepa2_encoder_ckpt to your local V-JEPA 2 encoder weights.

python3 \
    train.py --cfg cfgs/rae_vjepa.yaml \
    --manualSeed 66667 --tag default \
    --csv_file ucf101_train.csv+kinetics_dataset.csv --out_path save/vjepa_rae \
    --name larp_tokenizer -b 32 -j 32 \
    --frame_num 16 --input_size 256   \
    --opts \
    test_dataset.csv_paths.ucf101_val ucf101_val.csv \
    model.args.vjepa2_encoder_ckpt checkpoints/vitl.pt \
    model.args.bottleneck_token_num 1024 \
    model.args.encoder_hidden_size 768 \
    model.args.decoder_hidden_size 768 \
    model.args.encoder_depth 12 \
    model.args.decoder_depth 12 \
    model.args.encoder_num_heads 12 \
    model.args.decoder_num_heads 12 \
    loss.args.disc_tran_hidden_size 512 \
    loss.args.disc_tran_n_heads 8 \
    loss.args.disc_tran_n_layers 12 \
    optimizer.args.lr 0.0001  \
    optimizer.loss_args.lr 0.00001 \
    optimizer.warmup_epoch 8 \
    optimizer.min_lr_mult 0.01 \
    optimizer.prior_lr_mult 50.0 \
    optimizer.lr_type cosine \
    use_amp true \
    compile true \
    compile_mode default \
    vis_epoch 1 eval_epoch 1  max_epoch 150 latest_interval 1 save_best true 

# append --wandb-upload if you want to sync to wandb
# append --replace if you want to start a new training run instead of resuming from the latest checkpoint (if available)