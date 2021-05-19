#!/bin/bash

TASK=../experiment/dataset_fairseq
GPT2_BPE_DIR=../bart-tl/preprocess/bpe

for SPLIT in train test
do
  for LANG in source target
  do
    python3 "$(dirname "$0")"/multiprocessing_bpe_encoder.py \
    --encoder-json ${GPT2_BPE_DIR}/encoder.json \
    --vocab-bpe ${GPT2_BPE_DIR}/vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
