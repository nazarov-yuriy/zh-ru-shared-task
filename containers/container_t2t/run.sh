#!/bin/bash

mkdir -p /opt/results

sed 's~ ~~g' < /tmp/data/input.txt > input_wo_spaces.txt

t2t-decoder \
    --t2t_usr_dir /root/data \
    --data_dir /root/data \
    --problem translate_zhru_full \
    --model evolved_transformer \
    --hparams_set evolved_transformer_base \
    --output_dir /root/data \
    --decode_hparams="beam_size=1,alpha=0.6" \
    --decode_from_file=input_wo_spaces.txt \
    --decode_to_file=/opt/results/output.txt
