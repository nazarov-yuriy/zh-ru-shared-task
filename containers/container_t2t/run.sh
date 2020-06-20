#!/bin/bash

INPUT_FILE="/tmp/data/input.txt"
OUTPUT_DIR="/opt/results"
OUTPUT_FILE="/opt/results/output.txt"
TMP_FILE="input_wo_spaces.txt"
DATA_DIR="/root/data"

PROBLEM="translate_zhru_full"
MODEL="evolved_transformer"
HPARAMS_SET="evolved_transformer_base"
DECODE_HPARAMS="beam_size=1,alpha=0.6"

mkdir -p "$OUTPUT_DIR"

sed 's~ ~~g' < "$INPUT_FILE" > "$TMP_FILE"  # Fix undisclosed tokenization by removing spaces

t2t-decoder \
    --t2t_usr_dir "$DATA_DIR" \
    --data_dir "$DATA_DIR" \
    --problem "$PROBLEM" \
    --model "$MODEL" \
    --hparams_set "$HPARAMS_SET" \
    --output_dir "$DATA_DIR" \
    --decode_hparams "$DECODE_HPARAMS" \
    --decode_from_file "$TMP_FILE" \
    --decode_to_file "$OUTPUT_FILE"
