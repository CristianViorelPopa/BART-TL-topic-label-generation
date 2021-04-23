TOTAL_NUM_UPDATES=2916  # before: 20000
WARMUP_UPDATES=175  # before: 500
LR=3e-05  # before: 3e-05
MAX_TOKENS=2048  # before: 2048
UPDATE_FREQ=4  # before: 4
TASK=summarization/dataset_fairseq-bin
#TASK=summarization/models/bart_finetuned_terms_labels_v11/dataset_fairseq-bin
BART_PATH=summarization/finetune/models/bart.large/model.pt
#BART_PATH=summarization/models/bart_finetuned_terms_labels_v11/bart_finetuned_terms_labels_v11.pt
#BART_PATH=checkpoints/checkpoint10.pt

python3 "$(dirname "$0")"/train.py ${TASK} --restore-file ${BART_PATH} --max-tokens ${MAX_TOKENS} --task translation --source-lang source --target-lang target --truncate-source --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch bart_large --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr $LR --total-num-update ${TOTAL_NUM_UPDATES} --warmup-updates ${WARMUP_UPDATES} --update-freq ${UPDATE_FREQ} --skip-invalid-size-inputs-valid-test --find-unused-parameters --no-last-checkpoints;
