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
translate.py were inspired by https://cloud.google.com/translate/docs/basic/translating-text#translate_translate_text-python