Steps to reproduce in pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime docker image
```
apt update && apt install -y gcc g++ wget && pip install fairscale fairseq sentencepiece

wget https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model
wget https://dl.fbaipublicfiles.com/m2m_100/data_dict.128k.txt
wget https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt
wget https://dl.fbaipublicfiles.com/m2m_100/language_pairs_small_models.txt 
wget https://dl.fbaipublicfiles.com/m2m_100/1.2B_last_checkpoint.pt

tr -d ' ' < corpus.zh > corpus_no_spaces.zh
python3 fairseq/scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=corpus_no_spaces.zh --outputs=spm.zh-ru.zh
python3 fairseq/scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=corpus.ru --outputs=spm.zh-ru.ru
fairseq-preprocess --source-lang "zh" --target-lang "ru" --testpref spm.zh-ru --thresholdsrc 0 --thresholdtgt 0 --destdir data_bin_zh_ru --srcdict data_dict.128k.txt --tgtdict data_dict.128k.txt

fairseq-generate data_bin_zh_ru --batch-size 32 --path 1.2B_last_checkpoint.pt -s zh -t ru --remove-bpe 'sentencepiece' --beam 5 --task translation_multi_simple_epoch --lang-pairs language_pairs_small_models.txt --decoder-langtok --encoder-langtok src --gen-subset test --fixed-dictionary model_dict.128k.txt --fp16 > gen_out

cat gen_out | grep ^H | cut -f3- > corpus_m2m_100.ru
cat gen_out | grep ^T | cut -f2- > corpus_sorted.ru
cat corpus_m2m_100.ru | sacrebleu -l zh-ru corpus_sorted.ru
```
```
BLEU+case.mixed+lang.zh-ru+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 9.8 40.3/13.7/6.1/2.8 (BP = 1.000 ratio = 1.007 hyp_len = 122761 ref_len = 121901)
```
```
wget https://dl.fbaipublicfiles.com/m2m_100/418M_last_checkpoint.pt

fairseq-generate data_bin_zh_ru --batch-size 32 --path 418M_last_checkpoint.pt-s zh -t ru --remove-bpe 'sentencepiece' --beam 5 --task translation_multi_simple_epoch --lang-pairs language_pairs_small_models.txt --decoder-langtok --encoder-langtok src --gen-subset test --fixed-dictionary model_dict.128k.txt --fp16 > gen_out

cat gen_out | grep ^H | cut -f3- > corpus_m2m_100.ru
cat gen_out | grep ^T | cut -f2- > corpus_sorted.ru
cat corpus_m2m_100.ru | sacrebleu -l zh-ru corpus_sorted.ru
```

```
BLEU+case.mixed+lang.zh-ru+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 8.0 37.2/11.4/4.7/2.0 (BP = 1.000 ratio = 1.032 hyp_len = 125855 ref_len = 121901)
```