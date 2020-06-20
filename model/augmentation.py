import argparse
import re
import random
from multiprocessing import Pool
import fasttext
import hnswlib
import pymorphy2

datasets_real = [
    # http://opus.nlpl.eu/UNPC-v1.0.php
    ('../data/UNPC/ru-zh/UNPC.ru-zh.zh', '../data/UNPC/ru-zh/UNPC.ru-zh.ru'),
    # http://opus.nlpl.eu/TED2013-v1.1.php
    ('../data/TED2013-v1.1/zh-en-ru/TED2013.zh-en-ru.zh', '../data/TED2013-v1.1/zh-en-ru/TED2013.zh-en-ru.ru'),
    # http://opus.nlpl.eu/WMT-News-v2019.php
    ('../data/WMT-News-v2019/zh-en-ru/WMT-News.zh-en-ru.zh', '../data/WMT-News-v2019/zh-en-ru/WMT-News.zh-en-ru.ru'),
    # http://opus.nlpl.eu/OpenSubtitles-v2018.php
    ('../data/OpenSubtitles2018/ru-zh_cn/OpenSubtitles.ru-zh_cn.zh_cn',
     '../data/OpenSubtitles2018/ru-zh_cn/OpenSubtitles.ru-zh_cn.ru'),
    # my own
    ('../data/wikidata_titles/wikidata_titles.zh', '../data/wikidata_titles/wikidata_titles.ru'),
    # https://linguatools.org/tools/corpora/wikipedia-parallel-titles-corpora/
    ('../data/wikititles-2014_ruzh/wikititles-2014_ruzh.zh', '../data/wikititles-2014_ruzh/wikititles-2014_ruzh.ru'),
    # http://www.casmacat.eu/corpus/news-commentary.html
    ('../data/news-commentary/all.zh', '../data/news-commentary/all.ru'),
]

datasets_bt = [
    ('../data/news/news.zh', '../data/news/news.ru'),
    ('../data/proza/proza_ru_2017_01_short.zh', '../data/proza/proza_ru_2017_01_short.ru'),
    ('../data/proza/proza_ru_2017_02_short.zh', '../data/proza/proza_ru_2017_02_short.ru'),
    ('../data/proza/proza_ru_2017_03_short.zh', '../data/proza/proza_ru_2017_03_short.ru'),
    ('../data/proza/proza_ru_2017_04_short.zh', '../data/proza/proza_ru_2017_04_short.ru'),
    ('../data/proza/proza_ru_2017_05_short.zh', '../data/proza/proza_ru_2017_05_short.ru'),
    ('../data/proza/proza_ru_2017_06_short.zh', '../data/proza/proza_ru_2017_06_short.ru'),
    ('../data/proza/proza_ru_2017_07_short.zh', '../data/proza/proza_ru_2017_07_short.ru'),
    ('../data/proza/proza_ru_2017_08_short.zh', '../data/proza/proza_ru_2017_08_short.ru'),
    ('../data/proza/proza_ru_2017_09_short.zh', '../data/proza/proza_ru_2017_09_short.ru'),
    ('../data/proza/proza_ru_2017_10_short.zh', '../data/proza/proza_ru_2017_10_short.ru'),
    ('../data/proza/proza_ru_2017_11_short.zh', '../data/proza/proza_ru_2017_11_short.ru'),
    ('../data/proza/proza_ru_2017_12_short.zh', '../data/proza/proza_ru_2017_12_short.ru'),
]


def fix_moses(s):
    s = s.replace("&quot;", '"')
    s = s.replace("&lt; &lt; ", '«')
    s = s.replace(" &gt; &gt;", '»')
    s = s.replace("&lt;", '<')
    s = s.replace("&gt;", '>')
    s = s.replace("&amp;", '&')
    return s


def orig(gen_zh, gen_ru):
    for zh, ru in zip(gen_zh, gen_ru):
        zh = zh.rstrip()
        ru = ru.rstrip()
        if re.match(r'.*[а-яёА-ЯЁ]+', ru):
            yield fix_moses(zh), fix_moses(ru)


def lower(gen_zh, gen_ru):
    for zh, ru in zip(gen_zh, gen_ru):
        zh = zh.rstrip()
        ru = ru.rstrip()
        if re.match(r'.*[а-яёА-ЯЁ]+', ru):
            yield fix_moses(zh), fix_moses(ru).lower()


def upper_random(gen_zh, gen_ru):
    for zh, ru in zip(gen_zh, gen_ru):
        zh = zh.rstrip()
        ru = ru.rstrip()
        if re.match(r'.*[а-яёА-ЯЁ]+', ru):
            yield fix_moses(zh), ' '.join([x.upper() if random.random() < 0.2 else x for x in fix_moses(ru).split(' ')])


def remove_punct(gen_zh, gen_ru):
    for zh, ru in zip(gen_zh, gen_ru):
        zh = zh.rstrip()
        ru = ru.rstrip()
        if re.match(r'.*[а-яёА-ЯЁ]+', ru):
            ru = " ".join([x for x in re.split(r'[^а-яёА-ЯЁ0-9]+', ru) if x])
            yield fix_moses(zh), ru


def add_spaces(gen_zh, gen_ru):
    for zh, ru in zip(gen_zh, gen_ru):
        zh = zh.rstrip()
        ru = ru.rstrip()
        if re.match(r'.*[а-яёА-ЯЁ]+', ru):
            ru = fix_moses(ru)
            zh = fix_moses(zh)
            if random.random() < 0.5:
                ru = "".join([x + " " if random.random() < 0.05 else x for x in ru])
            else:
                zh = "".join([x + " " if random.random() < 0.2 else x for x in zh])
            yield zh, ru


def mask(gen_zh, gen_ru):
    for zh, ru in zip(gen_zh, gen_ru):
        zh = zh.rstrip()
        ru = ru.rstrip()
        if re.match(r'.*[а-яёА-ЯЁ]+', ru):
            ru = fix_moses(ru)
            zh = fix_moses(zh)
            if random.random() < 0.1:
                ru = "".join(["_" if random.random() < 0.05 else x for x in ru])
            else:
                zh = "".join(["_" if random.random() < 0.2 else x for x in zh])
            yield zh, ru


def typos(gen_zh, gen_ru):
    d = {
        'а': ['о'] * 2,
        'е': ['и', 'ё'],
        'ё': ['е', 'о'],
        'и': ['е', 'ы'],
        'й': ['и'] * 2,
        'о': ['ё', 'a'],
        'у': ['ю'] * 2,
        'э': ['е'] * 2,
        'ю': ['у'] * 2,
        'я': ['а'] * 2,
        'ы': ['и'] * 2,
        'б': ['п'] * 2,
        'п': ['б'] * 2,
        'в': ['ф'] * 2,
        'ф': ['в'] * 2,
        'з': ['с'] * 2,
        'с': ['з'] * 2,
        'д': ['т'] * 2,
        'т': ['д'] * 2,
        'ж': ['ш'] * 2,
        'ш': ['ж'] * 2,
        'г': ['к'] * 2,
        'к': ['г'] * 2,
    }
    pos = 0
    for zh, ru in zip(gen_zh, gen_ru):
        zh = zh.rstrip()
        ru = ru.rstrip()
        if re.match(r'.*[а-яёА-ЯЁ]+', ru):
            ru = fix_moses(ru)
            zh = fix_moses(zh)
            ru = "".join([d.get(x, [x, x])[pos % 2] if random.random() < 0.05 else x for x in ru])
            yield zh, ru


def shuffle_words(gen_zh, gen_ru):
    for zh, ru in zip(gen_zh, gen_ru):
        zh = zh.rstrip()
        ru = ru.rstrip()
        if re.match(r'.*[а-яёА-ЯЁ]+', ru):
            ru = fix_moses(ru)
            zh = fix_moses(zh)
            ru_words = ru.split(' ')
            for _ in range(len(ru_words) // 5):
                pos = random.randrange(len(ru_words) - 1)
                ru_words[pos], ru_words[pos + 1] = ru_words[pos + 1], ru_words[pos]
            yield zh, " ".join(ru_words)


def fasttext_syn(gen_zh, gen_ru):
    ftmodel = fasttext.load_model("/mnt/twelve/ml/models/cc.ru.300.bin")
    ftwords = ftmodel.get_words()
    index = hnswlib.Index(space='cosine', dim=ftmodel.get_dimension())
    index.load_index('../data/hnsw_index_500k.dat')

    for zh, ru in zip(gen_zh, gen_ru):
        zh = zh.rstrip()
        ru = ru.rstrip()
        if re.match(r'.*[а-яёА-ЯЁ]+', ru):
            ru = fix_moses(ru)
            ru_words = ru.split(' ')
            for i in range(len(ru_words)):
                if random.random() < 0.05:
                    nn = index.knn_query(ftmodel.get_word_vector(ru_words[i]), k=10)[0][0]
                    for j in nn:
                        if ftwords[j].lower() != ru_words[i].lower() and len(ftwords[j]) < 10:
                            ru_words[i] = ftwords[j]
                            break
            yield fix_moses(zh), ' '.join(ru_words)


def lemmatize(gen_zh, gen_ru):
    morph = pymorphy2.MorphAnalyzer()
    for zh, ru in zip(gen_zh, gen_ru):
        zh = zh.rstrip()
        ru = ru.rstrip()
        if re.match(r'.*[а-яёА-ЯЁ]+', ru):
            ru = fix_moses(ru)
            ru_words = ru.split(' ')
            for i in range(len(ru_words)):
                if random.random() < 0.2:
                    p = morph.parse(ru_words[i])
                    if p:
                        ru_words[i] = p[0].normal_form
            yield fix_moses(zh), ' '.join(ru_words)


modes = {f.__name__: f for f in [
    orig, lower, upper_random, remove_punct, add_spaces,
    mask, typos, shuffle_words, fasttext_syn, lemmatize
]}


def augmentate(arg):
    ipath_zh, ipath_ru, mode = arg
    random.seed(10)
    with open(ipath_zh, 'r') as fi_zh, open(ipath_ru, 'r') as fi_ru:
        opath_zh = '../data/augmentation/' + mode + '/' + ipath_zh.split('/')[-1].replace('_cn', '')
        opath_ru = '../data/augmentation/' + mode + '/' + ipath_ru.split('/')[-1].replace('_cn', '')
        with open(opath_zh, 'w') as fo_zh, open(opath_ru, 'w') as fo_ru:
            for (zh, ru) in modes[mode](fi_zh, fi_ru):
                print(zh, file=fo_zh)
                print(ru, file=fo_ru)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode')
    argparser.add_argument('--threads', type=int, default=8)
    args = argparser.parse_args()
    p = Pool(args.threads)
    print(p.map(augmentate, [(x[0], x[1], args.mode) for x in datasets_real + datasets_bt]))


if __name__ == '__main__':
    main()
