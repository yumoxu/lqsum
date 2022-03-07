#!/bin/sh
# Ablation study for 64: no Differentiability (we use argmax instead of Gumble Softmax)
# remove: --mask-z-tokens-with-gumbel-scores
# add: --mask-z-tokens-with-tags 

ARCH=guided_bart_large_with_tagging
TAGGING_HEAD=binary_classification

LOG_DIR=/home/lqsum/log/model_71-bpeTags


TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=3e-05

BATCH_SIZE=1
UPDATE_FREQ=16

MAX_TOKENS=640  # 2048
MAX_SRC_TOKENS=640
MAX_Z_TOKENS=640
BART_PATH=/home/lqsum/model/bart.large/model_${MAX_TOKENS}.pt

DATA_BIN=$1
SAVE_DIR=$2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py $DATA_BIN \
    --tensorboard-logdir $LOG_DIR \
    --restore-file $BART_PATH \
    --batch-size $BATCH_SIZE \
    --max-source-positions $MAX_SRC_TOKENS \
    --max-target-positions $MAX_SRC_TOKENS \
    --max-z-positions $MAX_Z_TOKENS \
    --task guided_translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch $ARCH \
    --tag-on-doc \
    --load-z-tags \
    --mask-z-tokens-with-tags \
    --fixed-tau 0.9 \
    --tagging-head-name $TAGGING_HEAD \
    --criterion tagging_generation_joint_criterion \
    --tag-coef 10.0 \
    --post-entropy-coef 0.1 \
    --oracle-anneal-end-value 0.5 \
    --norm-loss-in-sample \
    --label-smoothing 0.1 \
    --tag-label-smoothing 0.0 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $SAVE_DIR \
    --find-unused-parameters \
    --num-workers=0 \
    --memory-efficient-fp16 \
    --fp16-scale-tolerance=0.25 \
    --ddp-backend=no_c10d \
