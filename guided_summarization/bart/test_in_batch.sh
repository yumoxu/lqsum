PROJ_ROOT=/home/lqsum 

DATASET=xsum  # cnndm, xsum
MODEL_ID=19xs  # specify this
START_CKPT=3
END_CKPT=5  # specify this; usually 9 or 10

SRC=${PROJ_ROOT}/data/${DATASET}/test.source
DATA_BIN=${PROJ_ROOT}/data/${DATASET}_bin/source_guidance_with_stem_tags
MODEL_DIR=${PROJ_ROOT}/model/model_${MODEL_ID}-stemTags

for i in $(seq $START_CKPT $END_CKPT)
do
    RESULT_PATH=${PROJ_ROOT}/bart_out/model_${MODEL_ID}_${i}-stemTags-${DATASET}_test-source
    MODEL_NAME=checkpoint${i}.pt
    echo "Generate for the ${i}th checkpoint. Results will be saved in: ${RESULT_PATH}"
    python test.py $SRC $RESULT_PATH $MODEL_DIR $MODEL_NAME $DATA_BIN
done
