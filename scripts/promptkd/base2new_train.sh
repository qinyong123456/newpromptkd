#!/bin/bash

# custom config
DATA="/kaggle/working/PromptKD/datasets/data"
TRAINER=PromptKD

DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
SEED=$2

CFG=vit_b16_c2_ep20_batch8_4+4ctx
SHOTS=0

DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}

# fgvc_aircraft, oxford_flowers, dtd: KD_WEIGHT:200
# imagenet, caltech101, eurosat, food101, oxford_pets, stanford_cars, sun397, ucf101, KD_WEIGHT:1000

python train.py \
    --config-file configs/trainers/promptkd.yaml \  # 添加这一行
    --data-dir data \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.MODAL base2novel \
    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
    TRAINER.PROMPTKD.KD_WEIGHT 1000.0 \
    TRAINER.PROMPTKD.LOGIT_STANDARDIZATION True \
    TRAINER.PROMPTKD.ADAPTIVE_TEMPERATURE True \
    TRAINER.PROMPTKD.USE_DMIXER True \
    TRAINER.PROMPTKD.DMIXER_LAYERS 4 \
    TRAINER.PROMPTKD.MIXER_KERNEL 3 \
    TRAINER.PROMPTKD.DMIXER_LR 5e-5
