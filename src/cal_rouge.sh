#!/bin/sh

export CLASSPATH=/home/utils/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
PROJ_ROOT=/home/lqsum

HYP_NAME=$1
HYP_PATH=${PROJ_ROOT}/bart_out/${HYP_NAME}
HYP_TOKENIZED_PATH=${PROJ_ROOT}/bart_out/tokenized/${HYP_NAME}.tokenized

if [ ! -f ${HYP_TOKENIZED_PATH} ]; then
    echo "Tokenizing model output"
    cat ${HYP_PATH} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${HYP_TOKENIZED_PATH}
else
    echo "Found tokenized model output"
fi

if [ "$2" == 'cnndm' ]; then
    REF_PATH=${PROJ_ROOT}/data/cnndm/test.target

elif [ "$2" == 'wikicat' ]; then

    if [ "$3" == 'animal' ]; then
        REF_PATH=${PROJ_ROOT}/data/wikicat/animal.target
    elif [ "$3" == 'company' ]; then
        REF_PATH=${PROJ_ROOT}/data/wikicat/company.target
    else
        REF_PATH=${PROJ_ROOT}/data/wikicat/film.target
    fi

elif [ "$2" == 'wikiref' ]; then
    
    if [ "$3" == 'noMatch' ]; then
        REF_PATH=${PROJ_ROOT}/data/wikiref/test.noMatch.target
    elif [ "$3" == 'withMatch' ]; then
        REF_PATH=${PROJ_ROOT}/data/wikiref/test.withMatch.target
    else
        REF_PATH=${PROJ_ROOT}/data/wikiref/test.target
    fi

elif [ "$2" == 'debatepedia' ]; then
    
    if [ "$3" == 'noMatch' ]; then
        REF_PATH=${PROJ_ROOT}/data/debatepedia/test.noMatch.target
    elif [ "$3" == 'withMatch' ]; then
        REF_PATH=${PROJ_ROOT}/data/debatepedia/test.withMatch.target
    else
        REF_PATH=${PROJ_ROOT}/data/debatepedia/test.target
    fi

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

echo "ROUGE calculation for $2"
echo "REF file: $REF_TOKENIZED_PATH"
files2rouge ${HYP_TOKENIZED_PATH} ${REF_TOKENIZED_PATH} -a "-c 95 -r 1000 -n 2 -a" --ignore_empty
