PROJ_ROOT=/home/lqsum 

DATASET=cnndm  # cnndm, xsum
MODEL_ID=68  # specify this
DATA_NAME=bpeTags  # specify this: stemTags, bpeTags

START_CKPT=11
END_CKPT=11  # specify this; usually 9 or 10
AFFIX=""  # wo_gumble_noise, -confirm

MODEL_DIR=${PROJ_ROOT}/model/model_${MODEL_ID}-${DATA_NAME}
SRC=${PROJ_ROOT}/data/${DATASET}/test.source
GUIDANCE=${PROJ_ROOT}/data/${DATASET}_guide_source/wo_tags/test.source

if [ $DATA_NAME == 'stemTags' ]; then
    DATA_BIN=${PROJ_ROOT}/data/${DATASET}_bin/source_guidance_with_stem_tags
elif [ $DATA_NAME == 'bpeTags' ]; then
    DATA_BIN=${PROJ_ROOT}/data/${DATASET}_bin/source_guidance_with_bpe_lcs_tags
else
    exit 1;
fi

for i in $(seq $START_CKPT $END_CKPT)
do
    RESULT_PATH=${PROJ_ROOT}/bart_out/model_${MODEL_ID}_${i}-${DATA_NAME}-${DATASET}_test-source${AFFIX}
    
    MODEL_NAME=checkpoint${i}.pt
    echo "Generate for the ${i}th checkpoint..."
    echo "SRC: ${SRC}"
    echo "GUIDANCE: ${GUIDANCE}"
    echo "DATA_BIN: ${DATA_BIN}"
    echo "Save to: ${RESULT_PATH}"
    python z_test_with_estimated_tags.py $SRC $GUIDANCE $RESULT_PATH $MODEL_DIR $MODEL_NAME $DATA_BIN
done