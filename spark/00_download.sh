#!/bin/bash

set -x -e
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
HDFS_BASE="${HDFS_BASE:-hdfs://ryzen:9000/user/root/dataset}"

mkdir -p UNPC-v1.0
cd UNPC-v1.0
wget -c http://opus.nlpl.eu/download.php?f=UNPC/v1.0/moses/ru-zh.txt.zip -O ru-zh.txt.zip
unzip -o ru-zh.txt.zip
/opt/hadoop-3.2.1/bin/hdfs dfs -mkdir -p "$HDFS_BASE/opus/UNPC-v1.0"
/opt/hadoop-3.2.1/bin/hdfs dfs -put -f UNPC.ru-zh.{ru,zh} "$HDFS_BASE/opus/UNPC-v1.0/"
cd ..

mkdir -p TED2013-v1.1
cd TED2013-v1.1
wget -c http://opus.nlpl.eu/download.php?f=TED2013/v1.1/moses/en-zh.txt.zip -O en-zh.txt.zip
wget -c http://opus.nlpl.eu/download.php?f=TED2013/v1.1/moses/en-ru.txt.zip -O en-ru.txt.zip
unzip -o en-zh.txt.zip
unzip -o en-ru.txt.zip
/opt/hadoop-3.2.1/bin/hdfs dfs -mkdir -p "$HDFS_BASE/opus/TED2013-v1.1"
/opt/hadoop-3.2.1/bin/hdfs dfs -put -f TED2013.*.{ru,en,zh} "$HDFS_BASE/opus/TED2013-v1.1/"
cd ..

mkdir -p WMT-News-v2019
cd WMT-News-v2019
wget -c http://opus.nlpl.eu/download.php?f=WMT-News/v2019/moses/en-zh.txt.zip -O en-zh.txt.zip
wget -c http://opus.nlpl.eu/download.php?f=WMT-News/v2019/moses/en-ru.txt.zip -O en-ru.txt.zip
unzip -o en-zh.txt.zip
rm README LICENSE
unzip -o en-ru.txt.zip
/opt/hadoop-3.2.1/bin/hdfs dfs -mkdir -p "$HDFS_BASE/opus/WMT-News-v2019"
/opt/hadoop-3.2.1/bin/hdfs dfs -put -f WMT-News.*.{ru,en,zh} "$HDFS_BASE/opus/WMT-News-v2019/"
cd ..

mkdir -p OpenSubtitles-v2018
cd OpenSubtitles-v2018
wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/ru-zh_cn.txt.zip -O ru-zh_cn.txt.zip
unzip -o ru-zh_cn.txt.zip
# Original files are somehow cursed(makes zipWithIndex stuck) on ubuntu 20.04+hadoop-3.2.1+spark-3.0.0
# Files with double size works just fine.
mv OpenSubtitles.ru-zh_cn.ru OpenSubtitles.ru-zh_cn.ru.tmp
cat OpenSubtitles.ru-zh_cn.ru.tmp OpenSubtitles.ru-zh_cn.ru.tmp > OpenSubtitles.ru-zh_cn.ru
mv OpenSubtitles.ru-zh_cn.zh_cn OpenSubtitles.ru-zh_cn.zh_cn.tmp
cat OpenSubtitles.ru-zh_cn.zh_cn.tmp OpenSubtitles.ru-zh_cn.zh_cn.tmp > OpenSubtitles.ru-zh_cn.zh_cn
/opt/hadoop-3.2.1/bin/hdfs dfs -mkdir -p "$HDFS_BASE/opus/OpenSubtitles-v2018"
/opt/hadoop-3.2.1/bin/hdfs dfs -put -f OpenSubtitles.*.{ru,zh_cn} "$HDFS_BASE/opus/OpenSubtitles-v2018/"
cd ..

mkdir -p wikipedia-parallel-titles-corpora
cd wikipedia-parallel-titles-corpora
wget -c https://www.dropbox.com/s/jcbphinsu1oedtb/wikititles-2014_ruzh.tgz?dl=1 -O wikititles-2014_ruzh.tgz
tar -xf wikititles-2014_ruzh.tgz
/opt/hadoop-3.2.1/bin/hdfs dfs -mkdir -p "$HDFS_BASE/wikipedia-parallel-titles-corpora"
/opt/hadoop-3.2.1/bin/hdfs dfs -put -f wikititles-2014_ruzh.{ru,zh} "$HDFS_BASE/wikipedia-parallel-titles-corpora/"
cd ..

mkdir -p news-commentary
cd news-commentary
wget -c http://www.casmacat.eu/corpus/news-commentary/aligned.tgz -O aligned.tgz
tar -xf aligned.tgz
cat aligned/Russian-Chinese/Chinese/* > all.zh
cat aligned/Russian-Chinese/Russian/* > all.ru
/opt/hadoop-3.2.1/bin/hdfs dfs -mkdir -p "$HDFS_BASE/news-commentary"
/opt/hadoop-3.2.1/bin/hdfs dfs -put -f all.{ru,zh} "$HDFS_BASE/news-commentary/"
cd ..
