import random
import pickle
from multiprocessing import Pool
import os
import sys
import gc
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder

# from tensor2tensor.data_generators import generator_utils

EOS = "<EOS>"
EOS_ID = 1
MAX_LEN = 128


def gen(path_zh, path_ru):
    random.seed(hash(path_ru) % 1000)
    tokenizer = SubwordTextEncoder('../data/vocab.translate_zhru_full.47000.subwords')
    with open(path_zh, 'r') as fzh, open(path_ru, 'r') as fru:
        sample = {"inputs": [], "targets": []}
        for line_zh, line_ru in zip(fzh, fru):
            ids_zh = tokenizer.encode(line_zh.rstrip() + ' ')
            ids_ru = tokenizer.encode(line_ru.rstrip() + ' ')
            if 0 == len(sample["inputs"]) or (len(sample["inputs"]) + 1 + len(ids_zh) <= MAX_LEN
                                              and len(sample["targets"]) + 1 + len(ids_ru) <= MAX_LEN
                                              and random.random() < 0.5):
                sample["inputs"].extend(ids_zh)
                sample["targets"].extend(ids_ru)
            else:
                sample["inputs"] = sample["inputs"][:MAX_LEN-1] + [EOS_ID]
                sample["targets"] = sample["targets"][:MAX_LEN-1] + [EOS_ID]
                yield sample.copy()
                sample["inputs"] = ids_zh
                sample["targets"] = ids_ru
    if sample["inputs"]:
        sample["inputs"] = sample["inputs"][:MAX_LEN - 1] + [EOS_ID]
        sample["targets"] = sample["targets"][:MAX_LEN - 1] + [EOS_ID]
        yield sample.copy()


def encode_file(arg):
    path_zh, path_ru, path_pickle = arg
    # while
    # if os.path.exists(path_pickle):
    #     st = os.stat(path_pickle)
    #     if st.st_size > 0:
    #         return
    batch_num = 0
    batch = []
    for el in gen(path_zh, path_ru):
        batch.append(el)
        if len(batch) >= 200000:
            with open(path_pickle + '.' + str(batch_num), 'wb') as f:
                pickle.dump(batch, f)
            batch_num += 1
            batch = []
            gc.collect()
            print('.', end='')
            sys.stdout.flush()
    if batch:
        with open(path_pickle + '.' + str(batch_num), 'wb') as f:
            pickle.dump(batch, f)
        print('.', end='')
        sys.stdout.flush()


# head -n 9000000 UNPC.ru-zh.ru > UNPC_h9m.ru-zh.ru; tail -n 7930367 UNPC.ru-zh.ru > UNPC_t8m.ru-zh.ru; head -n 9000000 UNPC.ru-zh.zh > UNPC_h9m.ru-zh.zh; tail -n 7930367 UNPC.ru-zh.zh > UNPC_t8m.ru-zh.zh; rm UNPC.ru-zh.ru; rm UNPC.ru-zh.zh
def main():
    zh_files = [
        '../data/augmentation/orig/all.zh',
        '../data/augmentation/orig/TED2013.zh-en-ru.zh',
        '../data/augmentation/orig/WMT-News.zh-en-ru.zh',
        '../data/augmentation/orig/OpenSubtitles.ru-zh.zh',
        '../data/augmentation/orig/wikidata_titles.zh',
        '../data/augmentation/orig/wikititles-2014_ruzh.zh',
        '../data/augmentation/orig/proza_ru_2017_01_short.zh',
        '../data/augmentation/orig/news.zh',
        '../data/augmentation/orig/proza_ru_2017_02_short.zh',
        '../data/augmentation/orig/proza_ru_2017_03_short.zh',
        '../data/augmentation/orig/proza_ru_2017_04_short.zh',
        '../data/augmentation/orig/proza_ru_2017_05_short.zh',
        '../data/augmentation/orig/proza_ru_2017_06_short.zh',
        '../data/augmentation/orig/proza_ru_2017_07_short.zh',
        '../data/augmentation/orig/proza_ru_2017_08_short.zh',
        '../data/augmentation/orig/proza_ru_2017_09_short.zh',
        '../data/augmentation/orig/proza_ru_2017_10_short.zh',
        '../data/augmentation/orig/proza_ru_2017_11_short.zh',
        '../data/augmentation/orig/proza_ru_2017_12_short.zh',
        '../data/augmentation/orig/UNPC_h9m.ru-zh.zh',
        '../data/augmentation/orig/UNPC_t8m.ru-zh.zh',
        '../data/augmentation/lower/all.zh',
        '../data/augmentation/lower/TED2013.zh-en-ru.zh',
        '../data/augmentation/lower/WMT-News.zh-en-ru.zh',
        '../data/augmentation/lower/OpenSubtitles.ru-zh.zh',
        '../data/augmentation/lower/wikidata_titles.zh',
        '../data/augmentation/lower/wikititles-2014_ruzh.zh',
        '../data/augmentation/lower/proza_ru_2017_01_short.zh',
        '../data/augmentation/lower/news.zh',
        '../data/augmentation/lower/proza_ru_2017_02_short.zh',
        '../data/augmentation/lower/proza_ru_2017_03_short.zh',
        '../data/augmentation/lower/proza_ru_2017_04_short.zh',
        '../data/augmentation/lower/proza_ru_2017_05_short.zh',
        '../data/augmentation/lower/proza_ru_2017_06_short.zh',
        '../data/augmentation/lower/proza_ru_2017_07_short.zh',
        '../data/augmentation/lower/proza_ru_2017_08_short.zh',
        '../data/augmentation/lower/proza_ru_2017_09_short.zh',
        '../data/augmentation/lower/proza_ru_2017_10_short.zh',
        '../data/augmentation/lower/proza_ru_2017_11_short.zh',
        '../data/augmentation/lower/proza_ru_2017_12_short.zh',
        '../data/augmentation/lower/UNPC_h9m.ru-zh.zh',
        '../data/augmentation/lower/UNPC_t8m.ru-zh.zh',
        '../data/augmentation/upper_random/all.zh',
        '../data/augmentation/upper_random/TED2013.zh-en-ru.zh',
        '../data/augmentation/upper_random/WMT-News.zh-en-ru.zh',
        '../data/augmentation/upper_random/OpenSubtitles.ru-zh.zh',
        '../data/augmentation/upper_random/wikidata_titles.zh',
        '../data/augmentation/upper_random/proza_ru_2017_01_short.zh',
        '../data/augmentation/upper_random/wikititles-2014_ruzh.zh',
        '../data/augmentation/upper_random/news.zh',
        '../data/augmentation/upper_random/proza_ru_2017_02_short.zh',
        '../data/augmentation/upper_random/proza_ru_2017_03_short.zh',
        '../data/augmentation/upper_random/proza_ru_2017_04_short.zh',
        '../data/augmentation/upper_random/proza_ru_2017_05_short.zh',
        '../data/augmentation/upper_random/proza_ru_2017_06_short.zh',
        '../data/augmentation/upper_random/proza_ru_2017_07_short.zh',
        '../data/augmentation/upper_random/proza_ru_2017_08_short.zh',
        '../data/augmentation/upper_random/proza_ru_2017_09_short.zh',
        '../data/augmentation/upper_random/proza_ru_2017_10_short.zh',
        '../data/augmentation/upper_random/proza_ru_2017_11_short.zh',
        '../data/augmentation/upper_random/proza_ru_2017_12_short.zh',
        '../data/augmentation/upper_random/UNPC_h9m.ru-zh.zh',
        '../data/augmentation/upper_random/UNPC_t8m.ru-zh.zh',
        '../data/augmentation/remove_punct/all.zh',
        '../data/augmentation/remove_punct/wikidata_titles.zh',
        '../data/augmentation/remove_punct/OpenSubtitles.ru-zh.zh',
        '../data/augmentation/remove_punct/TED2013.zh-en-ru.zh',
        '../data/augmentation/remove_punct/WMT-News.zh-en-ru.zh',
        '../data/augmentation/remove_punct/wikititles-2014_ruzh.zh',
        '../data/augmentation/remove_punct/news.zh',
        '../data/augmentation/remove_punct/proza_ru_2017_01_short.zh',
        '../data/augmentation/remove_punct/proza_ru_2017_02_short.zh',
        '../data/augmentation/remove_punct/proza_ru_2017_03_short.zh',
        '../data/augmentation/remove_punct/proza_ru_2017_04_short.zh',
        '../data/augmentation/remove_punct/proza_ru_2017_05_short.zh',
        '../data/augmentation/remove_punct/proza_ru_2017_06_short.zh',
        '../data/augmentation/remove_punct/proza_ru_2017_07_short.zh',
        '../data/augmentation/remove_punct/proza_ru_2017_08_short.zh',
        '../data/augmentation/remove_punct/proza_ru_2017_09_short.zh',
        '../data/augmentation/remove_punct/proza_ru_2017_10_short.zh',
        '../data/augmentation/remove_punct/proza_ru_2017_11_short.zh',
        '../data/augmentation/remove_punct/proza_ru_2017_12_short.zh',
        '../data/augmentation/remove_punct/UNPC_h9m.ru-zh.zh',
        '../data/augmentation/remove_punct/UNPC_t8m.ru-zh.zh',
        '../data/augmentation/mask/all.zh',
        '../data/augmentation/mask/TED2013.zh-en-ru.zh',
        '../data/augmentation/mask/WMT-News.zh-en-ru.zh',
        '../data/augmentation/mask/OpenSubtitles.ru-zh.zh',
        '../data/augmentation/mask/wikidata_titles.zh',
        '../data/augmentation/mask/wikititles-2014_ruzh.zh',
        '../data/augmentation/mask/news.zh',
        '../data/augmentation/mask/proza_ru_2017_01_short.zh',
        '../data/augmentation/mask/proza_ru_2017_02_short.zh',
        '../data/augmentation/mask/proza_ru_2017_03_short.zh',
        '../data/augmentation/mask/proza_ru_2017_04_short.zh',
        '../data/augmentation/mask/proza_ru_2017_05_short.zh',
        '../data/augmentation/mask/proza_ru_2017_06_short.zh',
        '../data/augmentation/mask/proza_ru_2017_07_short.zh',
        '../data/augmentation/mask/proza_ru_2017_08_short.zh',
        '../data/augmentation/mask/proza_ru_2017_09_short.zh',
        '../data/augmentation/mask/proza_ru_2017_10_short.zh',
        '../data/augmentation/mask/proza_ru_2017_11_short.zh',
        '../data/augmentation/mask/proza_ru_2017_12_short.zh',
        '../data/augmentation/mask/UNPC_h9m.ru-zh.zh',
        '../data/augmentation/mask/UNPC_t8m.ru-zh.zh',
        '../data/augmentation/add_spaces/all.zh',
        '../data/augmentation/add_spaces/TED2013.zh-en-ru.zh',
        '../data/augmentation/add_spaces/WMT-News.zh-en-ru.zh',
        '../data/augmentation/add_spaces/OpenSubtitles.ru-zh.zh',
        '../data/augmentation/add_spaces/wikidata_titles.zh',
        '../data/augmentation/add_spaces/wikititles-2014_ruzh.zh',
        '../data/augmentation/add_spaces/proza_ru_2017_01_short.zh',
        '../data/augmentation/add_spaces/news.zh',
        '../data/augmentation/add_spaces/proza_ru_2017_02_short.zh',
        '../data/augmentation/add_spaces/proza_ru_2017_03_short.zh',
        '../data/augmentation/add_spaces/proza_ru_2017_04_short.zh',
        '../data/augmentation/add_spaces/proza_ru_2017_05_short.zh',
        '../data/augmentation/add_spaces/proza_ru_2017_06_short.zh',
        '../data/augmentation/add_spaces/proza_ru_2017_07_short.zh',
        '../data/augmentation/add_spaces/proza_ru_2017_08_short.zh',
        '../data/augmentation/add_spaces/proza_ru_2017_09_short.zh',
        '../data/augmentation/add_spaces/proza_ru_2017_10_short.zh',
        '../data/augmentation/add_spaces/proza_ru_2017_11_short.zh',
        '../data/augmentation/add_spaces/proza_ru_2017_12_short.zh',
        '../data/augmentation/add_spaces/UNPC_h9m.ru-zh.zh',
        '../data/augmentation/add_spaces/UNPC_t8m.ru-zh.zh',
        '../data/augmentation/typos/all.zh',
        '../data/augmentation/typos/TED2013.zh-en-ru.zh',
        '../data/augmentation/typos/WMT-News.zh-en-ru.zh',
        '../data/augmentation/typos/wikidata_titles.zh',
        '../data/augmentation/typos/OpenSubtitles.ru-zh.zh',
        '../data/augmentation/typos/wikititles-2014_ruzh.zh',
        '../data/augmentation/typos/news.zh',
        '../data/augmentation/typos/proza_ru_2017_01_short.zh',
        '../data/augmentation/typos/proza_ru_2017_02_short.zh',
        '../data/augmentation/typos/proza_ru_2017_03_short.zh',
        '../data/augmentation/typos/proza_ru_2017_04_short.zh',
        '../data/augmentation/typos/proza_ru_2017_05_short.zh',
        '../data/augmentation/typos/proza_ru_2017_06_short.zh',
        '../data/augmentation/typos/proza_ru_2017_07_short.zh',
        '../data/augmentation/typos/proza_ru_2017_08_short.zh',
        '../data/augmentation/typos/proza_ru_2017_09_short.zh',
        '../data/augmentation/typos/proza_ru_2017_10_short.zh',
        '../data/augmentation/typos/proza_ru_2017_11_short.zh',
        '../data/augmentation/typos/proza_ru_2017_12_short.zh',
        '../data/augmentation/typos/UNPC_h9m.ru-zh.zh',
        '../data/augmentation/typos/UNPC_t8m.ru-zh.zh',
        '../data/augmentation/shuffle_words/all.zh',
        '../data/augmentation/shuffle_words/TED2013.zh-en-ru.zh',
        '../data/augmentation/shuffle_words/WMT-News.zh-en-ru.zh',
        '../data/augmentation/shuffle_words/OpenSubtitles.ru-zh.zh',
        '../data/augmentation/shuffle_words/wikidata_titles.zh',
        '../data/augmentation/shuffle_words/wikititles-2014_ruzh.zh',
        '../data/augmentation/shuffle_words/proza_ru_2017_01_short.zh',
        '../data/augmentation/shuffle_words/news.zh',
        '../data/augmentation/shuffle_words/proza_ru_2017_02_short.zh',
        '../data/augmentation/shuffle_words/proza_ru_2017_03_short.zh',
        '../data/augmentation/shuffle_words/proza_ru_2017_04_short.zh',
        '../data/augmentation/shuffle_words/proza_ru_2017_05_short.zh',
        '../data/augmentation/shuffle_words/proza_ru_2017_06_short.zh',
        '../data/augmentation/shuffle_words/proza_ru_2017_09_short.zh',
        '../data/augmentation/shuffle_words/proza_ru_2017_07_short.zh',
        '../data/augmentation/shuffle_words/proza_ru_2017_08_short.zh',
        '../data/augmentation/shuffle_words/proza_ru_2017_10_short.zh',
        '../data/augmentation/shuffle_words/proza_ru_2017_11_short.zh',
        '../data/augmentation/shuffle_words/proza_ru_2017_12_short.zh',
        '../data/augmentation/shuffle_words/UNPC_h9m.ru-zh.zh',
        '../data/augmentation/shuffle_words/UNPC_t8m.ru-zh.zh',
        '../data/augmentation/fasttext_syn/all.zh',
        '../data/augmentation/fasttext_syn/OpenSubtitles.ru-zh.zh',
        '../data/augmentation/fasttext_syn/wikidata_titles.zh',
        '../data/augmentation/fasttext_syn/wikititles-2014_ruzh.zh',
        '../data/augmentation/fasttext_syn/news.zh',
        '../data/augmentation/fasttext_syn/proza_ru_2017_01_short.zh',
        '../data/augmentation/fasttext_syn/proza_ru_2017_02_short.zh',
        '../data/augmentation/fasttext_syn/proza_ru_2017_03_short.zh',
        '../data/augmentation/fasttext_syn/proza_ru_2017_04_short.zh',
        '../data/augmentation/fasttext_syn/proza_ru_2017_05_short.zh',
        '../data/augmentation/fasttext_syn/proza_ru_2017_06_short.zh',
        '../data/augmentation/fasttext_syn/proza_ru_2017_07_short.zh',
        '../data/augmentation/fasttext_syn/TED2013.zh-en-ru.zh',
        '../data/augmentation/fasttext_syn/proza_ru_2017_08_short.zh',
        '../data/augmentation/fasttext_syn/WMT-News.zh-en-ru.zh',
        '../data/augmentation/fasttext_syn/proza_ru_2017_09_short.zh',
        '../data/augmentation/fasttext_syn/proza_ru_2017_12_short.zh',
        '../data/augmentation/fasttext_syn/proza_ru_2017_10_short.zh',
        '../data/augmentation/fasttext_syn/proza_ru_2017_11_short.zh',
        '../data/augmentation/fasttext_syn/UNPC_h9m.ru-zh.zh',
        '../data/augmentation/fasttext_syn/UNPC_t8m.ru-zh.zh',
        '../data/augmentation/lemmatize/all.zh',
        '../data/augmentation/lemmatize/news.zh',
        '../data/augmentation/lemmatize/proza_ru_2017_01_short.zh',
        '../data/augmentation/lemmatize/OpenSubtitles.ru-zh.zh',
        '../data/augmentation/lemmatize/TED2013.zh-en-ru.zh',
        '../data/augmentation/lemmatize/WMT-News.zh-en-ru.zh',
        '../data/augmentation/lemmatize/wikidata_titles.zh',
        '../data/augmentation/lemmatize/wikititles-2014_ruzh.zh',
        '../data/augmentation/lemmatize/proza_ru_2017_02_short.zh',
        '../data/augmentation/lemmatize/proza_ru_2017_03_short.zh',
        '../data/augmentation/lemmatize/proza_ru_2017_04_short.zh',
        '../data/augmentation/lemmatize/proza_ru_2017_05_short.zh',
        '../data/augmentation/lemmatize/proza_ru_2017_06_short.zh',
        '../data/augmentation/lemmatize/proza_ru_2017_07_short.zh',
        '../data/augmentation/lemmatize/proza_ru_2017_08_short.zh',
        '../data/augmentation/lemmatize/proza_ru_2017_09_short.zh',
        '../data/augmentation/lemmatize/proza_ru_2017_10_short.zh',
        '../data/augmentation/lemmatize/proza_ru_2017_11_short.zh',
        '../data/augmentation/lemmatize/proza_ru_2017_12_short.zh',
        '../data/augmentation/lemmatize/UNPC_h9m.ru-zh.zh',
        '../data/augmentation/lemmatize/UNPC_t8m.ru-zh.zh',
    ]
    tuples = [
        (x, x[:-3] + '.ru', x[:-3] + '.pickle') for x in zh_files
    ]
    p = Pool(16)
    print(p.map(encode_file, tuples))


if __name__ == '__main__':
    main()
