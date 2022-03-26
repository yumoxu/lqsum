# LQSum
This repository releases the code for  Document Summarization with Latent Queries.

Please cite the following paper if you use this code,

> The availability of large-scale datasets has driven the development of neural models that create summaries from single documents, for generic purposes. When using a summarization system, users often have specific intents with various language realizations, which, depending on the information need, can range from a single keyword to a long narrative composed of multiple questions. Existing summarization systems, however, often either fail to support or act robustly on this query focused summarization task. We introduce LQSum, the first unified text summarization system that learns Latent Queries from documents for abstractive summarization with any existing query forms. Under a deep generative framework, our system jointly optimizes a latent query model and a conditional language model, allowing users to plug-and-play queries of any type at test time. Despite learning from only generic summarization data and requiring no further optimization for downstream summarization tasks, our system robustly outperforms strong comparison systems across summarization benchmarks with different query types, document settings, and target domains.

Should you have any query please contact me at [yumo.xu@ed.ac.uk](mailto:mailto:yumo.xu@ed.ac.uk).


## Data Construction
Our model operates over BPEs. 
To build data for training and testing, the first step is to do BPE.

### BPE
1. For our training set, CNN/DM, we need to BPE source document and target summary (as query is not accessible). The following script should be run for `IO=[source|target]`:

```bash
# cnndm: train
DATASET=train && IO=source && RAW=~/lqsum/data/cnndm/${DATASET}.${IO} && TGT=~/lqsum/data/cnndm_bpe/origin/${DATASET}.bpe.${IO} && cd ~/lqsum/guided_summarization/bart && . z_bpe.sh ${RAW} ${TGT}
# cnndm: val
DATASET=val && IO=source && RAW=~/lqsum/data/cnndm/${DATASET}.${IO} && TGT=~/lqsum/data/cnndm_bpe/origin/${DATASET}.bpe.${IO} && cd ~/lqsum/guided_summarization/bart && . z_bpe.sh ${RAW} ${TGT}
# cnndm: test
DATASET=test && IO=source && RAW=~/lqsum/data/cnndm/${DATASET}.${IO} && TGT=~/lqsum/data/cnndm_bpe/origin/${DATASET}.bpe.${IO} && cd ~/lqsum/guided_summarization/bart && . z_bpe.sh ${RAW} ${TGT}
```

2. For our test set, we need to BPE source document and query (as target summary is not available). The following script should be run for `IO=[source|query]`:

```bash
# wikiref
DATASET=test && IO=source && RAW=~/lqsum/data/wikiref/${DATASET}.${IO} && TGT=~/lqsum/data/wikiref_bpe/origin/${DATASET}.bpe.${IO} && cd ~/lqsum/guided_summarization/bart && . z_bpe.sh ${RAW} ${TGT}
# duc
DATASET=2007 && IO=source && RAW=~/lqsum/data/duc/duc_${DATASET}.ranked.lines/duc_${DATASET}.${IO} && TGT=~/lqsum/data/duc_bpe/origin/duc_${DATASET}.bpe.${IO} && cd ~/lqsum/guided_summarization/bart && . z_bpe.sh ${RAW} ${TGT}
# tdqfs
DATASET=tdqfs && IO=source && RAW=~/lqsum/data/${DATASET}/${DATASET}.ranked.lines/${DATASET}.${IO} && TGT=~/lqsum/data/${DATASET}_bpe/qe/${DATASET}.bpe.source && cd ~/lqsum/guided_summarization/bart && . z_bpe.sh ${RAW} ${TGT}
```

### Extract Tags
```bash
cd ~/lqsum/bottom_up && DATASET=train && BPE_ROOT=~/lqsum/data/cnndm_bpe/origin && GUID=${BPE_ROOT}/${DATASET}.bpe.source && TGT=${BPE_ROOT}/${DATASET}.bpe.target && OUTPUT=~/lqsum/data/cnndm_guide_source/with_bpe_lcs_tags/${DATASET}.source.bu && python preprocess_copy_bpe.py -src $GUID -tgt $TGT -output $OUTPUT
```
Replace `cnndm_bpe` with, e.g., `wikiref_bpe` for test sets.

### Binarize 
Training data need to be binarized for efficieny training. 

We first move the data to be binarized from `data/cnndm_guide_source` to `data/cnndm_bpe`. Use the following command for `DATASET=[train|val|test]`:

```bash
DATASET=train && GUID_ROOT=~/lqsum/data/cnndm_guide_source/with_bpe_lcs_tags/${DATASET}.source.bu && BPE_ROOT=~/lqsum/data/cnndm_bpe/source_guidance_with_bpe_lcs_tags && cp ${GUID_ROOT}/${DATASET}.source.bu.src ${BPE_ROOT}/${DATASET}.bpe.z && cp ${GUID_ROOT}/${DATASET}.source.bu.tgt ${BPE_ROOT}/${DATASET}.bpe.tag
```

Then we binarize the data and save it to `data/cnndm_bin`:
```bash
cd ~/lqsum/guided_summarization/bart && . z_bin_with_tags.sh ~/lqsum/data/cnndm_bpe/source_guidance_with_bpe_lcs_tags ~/lqsum/data/cnndm_bin/source_guidance_with_bpe_lcs_tags
```

## Training
LQSum is developed on the basis of [GSum](https://github.com/neulab/guided_summarization), and some codes under `lqsum/guided_summarization` are borrowed from the original implementation of GSum. We thank the authors for their great work!

```bash
cd ~/lqsum/guided_summarization/bart && sh lqsum_train_main.sh ~/lqsum/data/cnndm_bin/source_guidance_with_bpe_lcs_tags ~/lqsum/model_[MODEL-ID]/
```
To train models for ablation study, replace `lqsum_train_main.sh` with `lqsum_train_ablation_[ABLATION-NAME].sh`, with `ABLATION-NAME` from `{dual_view, joint_training, posterior_dropout, weak_supervision}`.

## Decoding
Following are the commands for summary decoding on various test sets: WikiRef, Debatepedia, DUC, and TD-QFS. 
We set `TAG_MODE=query_11` to inject query into decoding via belief update.

```bash
# wikiref
DATASET=wikiref && MODEL_ID=64 && CKPT=10 && GUID_NAME=with_bpe_lcs_tags && TAG_MODE=query_11 && MODEL_DIR=~/lqsum/model/model_${MODEL_ID}-bpeTags && MODEL_NAME=checkpoint${CKPT}.pt && BART_OUT_NAME=model_${MODEL_ID}_${CKPT}-bpeTags-wikiref_test-source-${GUID_NAME}.${TAG_MODE}.min35max90 && SRC=~/lqsum/data/${DATASET}/test.source && GUIDANCE=~/lqsum/data/${DATASET}_prior/${GUID_NAME}/test.source.bu/test.source.bu.txt && RESULT_PATH=~/lqsum/bart_out/${BART_OUT_NAME} && DATA_BIN=~/lqsum/data/cnndm_bin/source_guidance_with_bpe_lcs_tags && CUDA_VISIBLE_DEVICES=0,1 . z_test_with_query_tags.sh $SRC $GUIDANCE $RESULT_PATH $MODEL_DIR $MODEL_NAME $DATA_BIN $TAG_MODE
# debatepedia
DATASET=debatepedia && MODEL_ID=64 && CKPT=10 && GUID_NAME=with_bpe_lcs_tags && TAG_MODE=query_11 && MODEL_DIR=~/lqsum/model/model_${MODEL_ID}-bpeTags && MODEL_NAME=checkpoint${CKPT}.pt && BART_OUT_NAME=model_${MODEL_ID}_${CKPT}-bpeTags-${DATASET}_test-source-${GUID_NAME}.${TAG_MODE}.min5max25 && SRC=~/lqsum/data/${DATASET}/test.source && GUIDANCE=~/lqsum/data/${DATASET}_prior/${GUID_NAME}/test.source.bu/test.source.bu.txt && RESULT_PATH=~/lqsum/bart_out/${BART_OUT_NAME} && DATA_BIN=~/lqsum/data/cnndm_bin/source_guidance_with_bpe_lcs_tags && CUDA_VISIBLE_DEVICES=0,1 . z_test_with_query_tags.sh $SRC $GUIDANCE $RESULT_PATH $MODEL_DIR $MODEL_NAME $DATA_BIN $TAG_MODE
# duc
DATASET=duc_2007 && MODEL_ID=64 && CKPT=10 && GUID_NAME=with_bpe_lcs_tags_marge && TAG_MODE=query_11 && MIN_MAX=min300max400 && MODEL_DIR=~/lqsum/model/model_${MODEL_ID}-bpeTags && MODEL_NAME=checkpoint${CKPT}.pt && BART_OUT_NAME=model_${MODEL_ID}_${CKPT}-bpeTags-${DATASET}-source-${GUID_NAME}.${TAG_MODE}.${MIN_MAX} && SRC=~/lqsum/data/duc/${DATASET}.marge.lines/${DATASET}.source && GUIDANCE=~/lqsum/data/duc_prior/${GUID_NAME}/${DATASET}.source.bu/${DATASET}.source.bu.txt && RESULT_PATH=~/lqsum/bart_out/${BART_OUT_NAME} && DATA_BIN=~/lqsum/data/cnndm_bin/source_guidance_with_bpe_lcs_tags && CUDA_VISIBLE_DEVICES=0,1 . z_test_with_query_tags.sh $SRC $GUIDANCE $RESULT_PATH $MODEL_DIR $MODEL_NAME $DATA_BIN $TAG_MODE
# tdqfs
DATASET=tdqfs && MODEL_ID=64 && CKPT=10 && GUID_NAME=with_bpe_lcs_tags_and_qe_3 && TAG_MODE=query_11 && MIN_MAX=min10max60 && MODEL_DIR=~/lqsum/model/model_${MODEL_ID}-bpeTags && MODEL_NAME=checkpoint${CKPT}.pt && BART_OUT_NAME=model_${MODEL_ID}_${CKPT}-bpeTags-${DATASET}-source-${GUID_NAME}.${TAG_MODE}.${MIN_MAX} && SRC=~/lqsum/data/tdqfs/${DATASET}.ranked.lines/${DATASET}.source && GUIDANCE=~/lqsum/data/${DATASET}_prior/${GUID_NAME}/${DATASET}.source.bu/${DATASET}.source.bu.txt && RESULT_PATH=~/lqsum/bart_out/${BART_OUT_NAME} && DATA_BIN=~/lqsum/data/cnndm_bin/source_guidance_with_bpe_lcs_tags && CUDA_VISIBLE_DEVICES=0,1 . z_test_with_query_tags.sh $SRC $GUIDANCE $RESULT_PATH $MODEL_DIR $MODEL_NAME $DATA_BIN $TAG_MODE
```

## Evaluation
Use the following command of evaluating summaries for WikiRef:

```bash
MODEL_ID=64 && CKPT=10 TAG_MODE=query_11 && sh src/cal_rouge.sh model_${MODEL_ID}_${CKPT}-bpeTags-wikiref_test-source-with_bpe_lcs_tags.${TAG_MODE}.min35max90 wikiref
```

For other test sets:
1. Replace `wikiref` as with `{debatepedia, duc2006, duc2007, tdqfs}` 
2. Change the second parameter to the corresponding summary output path