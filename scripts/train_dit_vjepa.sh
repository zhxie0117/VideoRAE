#!/usr/bin/env bash
# Train a 1D DiT on frozen continuous VideoRAE latents (8 GPUs by default).
# Edit AE_CKPT / RESULTS_DIR before running.

AE_CKPT="checkpoints/videorae_continuous.pth"
DATA_PATH="data/metadata"
CSV_FILE="ucf101_train.csv"
RESULTS_DIR="save/dit_vjepa"

NUM_GPUS=8
MODEL="DiT1D-XL"
NOISE_SCHEDULE="linear"
DIFFUSION_STEPS=1000
SAMPLING_STEPS=250

NUM_CLASSES=101
CLASS_DROPOUT_PROB=0.1
CFG_SCALE=2
EPOCHS=3000
BATCH_SIZE=64
LR=1e-4
MIXED_PRECISION="bf16"

FRAME_NUM=16
IMAGE_SIZE=256

torchrun --nproc_per_node=${NUM_GPUS} \
    latent_diffusion/train.py \
    --ae-ckpt ${AE_CKPT} \
    --data-path ${DATA_PATH} \
    --csv-file ${CSV_FILE} \
    --results-dir ${RESULTS_DIR} \
    --model ${MODEL} \
    --noise-schedule ${NOISE_SCHEDULE} \
    --diffusion-steps ${DIFFUSION_STEPS} \
    --sampling-steps ${SAMPLING_STEPS} \
    --num-classes ${NUM_CLASSES} \
    --class-dropout-prob ${CLASS_DROPOUT_PROB} \
    --cfg-scale ${CFG_SCALE} \
    --epochs ${EPOCHS} \
    --global-batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --mixed-precision ${MIXED_PRECISION} \
    --frame-num ${FRAME_NUM} \
    --image-size ${IMAGE_SIZE} \
    --max-grad-norm 1.0 \
    --log-every 50 \
    --ckpt-every 30000 \
    --sample-every 30000 \
    --num-workers 32
