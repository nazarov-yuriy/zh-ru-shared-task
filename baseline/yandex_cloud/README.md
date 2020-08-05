Steps to reproduce in ubuntu 18.04 docker image
```
apt update
apt install curl
curl https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash
yc init
export FOLDER_ID= #id after https://console.cloud.yandex.ru/folders/
export IAM_TOKEN=`yc iam create-token`

apt install python3 python3-pip
pip3 install sacrebleu
tr -d ' ' < corpus.zh > corpus_no_spaces.zh
mkdir ru zh
(cd zh; split -l 100 ../corpus_no_spaces.zh -d)
for i in {00..62}; do echo $i; LANG=C.UTF-8 python3 translate.py --src zh/x$i --dst ru/x$i --folder_id $FOLDER_ID; done
cat ru/x{00..62} > corpus_yandex.ru
cat corpus_yandex.ru | sacrebleu -l zh-ru corpus.ru
```
```
BLEU+case.mixed+lang.zh-ru+numrefs.1+smooth.exp+tok.13a+version.1.4.13 = 10.8 41.5/14.6/6.7/3.3 (BP = 0.998 ratio = 0.998 hyp_len = 121479 ref_len = 121709)
```
translate.py were inspired by https://cloud.yandex.ru/docs/translate/operations/translate