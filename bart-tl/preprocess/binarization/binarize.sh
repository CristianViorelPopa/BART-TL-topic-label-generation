#!/bin/bash

TASK=../experiment/dataset_fairseq
GPT2_DICT_DIR=../bart-tl/preprocess/bpe

python3 "$(dirname "$0")"/preprocess.py \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/test.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 60 \
  --srcdict ${GPT2_DICT_DIR}/dict.txt \
  --tgtdict ${GPT2_DICT_DIR}/dict.txt;
