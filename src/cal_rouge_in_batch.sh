#!/bin/sh
export CLASSPATH=/home/utils/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
PROJ_ROOT=/home/lqsum
DATASET='cnndm'

if [ $DATASET == 'cnndm' ]; then
    REF_PATH=${PROJ_ROOT}/data/cnndm/test.target
elif [ $DATASET == 'xsum' ]; then
    REF_PATH=${PROJ_ROOT}/data/xsum/test.target
elif [ $DATASET == 'wikiref' ]; then
    REF_PATH=${PROJ_ROOT}/data/wikiref/test.tgt
elif [ $DATASET == 'debatepedia' ]; then
    REF_PATH=${PROJ_ROOT}/data/debatepedia/test_summary
else
    exit 1;
fi

REF_TOKENIZED_PATH=${REF_PATH}.tokenized
if [ ! -f ${REF_TOKENIZED_PATH} ]; then
    echo "Tokenizing reference"
    cat ${REF_PATH} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${REF_TOKENIZED_PATH}
else
    echo "Found tokenized reference"
fi

echo "ROUGE calculation for $DATASET"

MODEL_ID=68  # specify this
START_CKPT=11  # specify this; usually 5 or 6
END_CKPT=11  # specify this; usually 9 or 10
DATA_NAME=bpeTags  # specify this: stemTags, bpeTags

ROUGE_SAVE_NAME=model_${MODEL_ID}-${DATA_NAME}-${DATASET}_test-source.rouge  # specify this
ROUGE_SAVE_PATH=${PROJ_ROOT}/bart_out/${ROUGE_SAVE_NAME}

TEMP_ROUGE_SAVE_PATH=${ROUGE_SAVE_PATH}.temp

for i in $(seq $START_CKPT $END_CKPT)  # for i in {5..10}
do
    HYP_NAME=model_${MODEL_ID}_${i}-${DATA_NAME}-${DATASET}_test-source  # specify this
    HYP_PATH=${PROJ_ROOT}/bart_out/${HYP_NAME}
    HYP_TOKENIZED_PATH=${PROJ_ROOT}/bart_out/tokenized/${HYP_NAME}.tokenized

    # if [ ! -f ${HYP_TOKENIZED_PATH} ]; then  # make hyp tokenization optional
    #     echo "Tokenizing model output"
    #     cat ${HYP_PATH} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${HYP_TOKENIZED_PATH}
    # else
    #     echo "Found tokenized model output"
    # fi

    if [ -f ${HYP_TOKENIZED_PATH} ]; then  # remove existing tokenization
        echo "Found tokenized model output, remove?"
        rm ${HYP_TOKENIZED_PATH}
    else
        echo "Tokenizing model output"
    fi
        
    cat ${HYP_PATH} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${HYP_TOKENIZED_PATH}
    echo "Finished tokenization. Calculate ROUGE..."

    echo "${i}th Checkpoint: " >> ${ROUGE_SAVE_PATH}
    files2rouge ${HYP_TOKENIZED_PATH} ${REF_TOKENIZED_PATH} -a "-c 95 -r 1000 -n 2 -a" -s ${TEMP_ROUGE_SAVE_PATH}

    rouge_result=`cat ${TEMP_ROUGE_SAVE_PATH}`
    echo "$rouge_result" >> ${ROUGE_SAVE_PATH}
done

rm -f TEMP_ROUGE_SAVE_PATH