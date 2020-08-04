Steps to reproduce in ubuntu 18.04 docker image
```
apt update
apt install python3 python3-pip
pip3 install --user --upgrade google-cloud-translate
pip3 install sacrebleu
tr -d ' ' < corpus.zh > corpus_no_spaces.zh
mkdir ru zh
(cd zh; split -l 100 ../corpus_no_spaces.zh -d)
for i in {00..62}; do LANG=C.UTF-8 GOOGLE_APPLICATION_CREDENTIALS=... python3 translate.py --src=zh/x$i --dst=ru/x$i --project_id=...; done
cat ru/x{00..62} > corpus_google.ru
cat corpus_google.ru | sacrebleu -l zh-ru corpus.ru
```
```
BLEU+case.mixed+lang.zh-ru+numrefs.1+smooth.exp+tok.13a+version.1.4.13 = 9.8 39.9/13.6/6.0/2.8 (BP = 1.000 ratio = 1.041 hyp_len = 126685 ref_len = 121709)
```
translate.py were inspired by https://cloud.google.com/translate/docs/basic/translating-text#translate_translate_text-python