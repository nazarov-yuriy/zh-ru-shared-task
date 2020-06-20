from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor.models import evolved_transformer

_TRAIN_DATA_SOURCES = [
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/UNPC/ru-zh/UNPC.ru-zh.h17900000.tgz",
        "input": "UNPC.ru-zh.h17900000.zh",
        "target": "UNPC.ru-zh.h17900000.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/TED_and_News.en-zh.tgz",
        "input": "TED_and_News.en-zh.zh",
        "target": "TED_and_News.en-zh.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/OpenSubtitles2018/ru-zh_cn/OpenSubtitles.ru-zh_cn.tgz",
        "input": "OpenSubtitles.ru-zh_cn.zh_cn",
        "target": "OpenSubtitles.ru-zh_cn.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/wikidata_titles.tgz",
        "input": "wikidata_titles.zh",
        "target": "wikidata_titles.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/wikititles-2014_ruzh.tgz",
        "input": "wikititles-2014_ruzh.zh",
        "target": "wikititles-2014_ruzh.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/news/news.tgz",
        "input": "news.zh",
        "target": "news.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/proza/proza_ru_2017_01_short.tgz",
        "input": "proza_ru_2017_01_short.zh",
        "target": "proza_ru_2017_01_short.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/proza/proza_ru_2017_02_short.tgz",
        "input": "proza_ru_2017_02_short.zh",
        "target": "proza_ru_2017_02_short.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/proza/proza_ru_2017_03_short.tgz",
        "input": "proza_ru_2017_03_short.zh",
        "target": "proza_ru_2017_03_short.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/proza/proza_ru_2017_04_short.tgz",
        "input": "proza_ru_2017_04_short.zh",
        "target": "proza_ru_2017_04_short.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/proza/proza_ru_2017_05_short.tgz",
        "input": "proza_ru_2017_05_short.zh",
        "target": "proza_ru_2017_05_short.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/proza/proza_ru_2017_06_short.tgz",
        "input": "proza_ru_2017_06_short.zh",
        "target": "proza_ru_2017_06_short.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/proza/proza_ru_2017_07_short.tgz",
        "input": "proza_ru_2017_07_short.zh",
        "target": "proza_ru_2017_07_short.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/proza/proza_ru_2017_08_short.tgz",
        "input": "proza_ru_2017_08_short.zh",
        "target": "proza_ru_2017_08_short.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/proza/proza_ru_2017_09_short.tgz",
        "input": "proza_ru_2017_09_short.zh",
        "target": "proza_ru_2017_09_short.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/proza/proza_ru_2017_10_short.tgz",
        "input": "proza_ru_2017_10_short.zh",
        "target": "proza_ru_2017_10_short.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/proza/proza_ru_2017_11_short.tgz",
        "input": "proza_ru_2017_11_short.zh",
        "target": "proza_ru_2017_11_short.ru",
    },
    {
        "url": "file:///mnt/nvm/ml/dataset/mlbootcamp/proza/proza_ru_2017_12_short.tgz",
        "input": "proza_ru_2017_12_short.zh",
        "target": "proza_ru_2017_12_short.ru",
    },
]
_ENDE_TRAIN_DATASETS = [
    [x["url"], (x["input"], x["target"])] for x in _TRAIN_DATA_SOURCES
]

_ENDE_EVAL_DATASETS = [
    [
        "file:///mnt/nvm/ml/dataset/mlbootcamp/UNPC/ru-zh/UNPC.ru-zh.t10000.tgz",
        ("UNPC.ru-zh.t10000.zh", "UNPC.ru-zh.t10000.ru")
    ],
]


@registry.register_problem
class TranslateZhru(translate.TranslateProblem):
    """En-de translation trained on WMT corpus."""

    @property
    def additional_training_datasets(self):
        """Allow subclasses to add training datasets."""
        return []

    def source_data_files(self, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        train_datasets = _ENDE_TRAIN_DATASETS + self.additional_training_datasets
        return train_datasets if train else _ENDE_EVAL_DATASETS

    @property
    def approx_vocab_size(self):
        return 48000


@registry.register_problem
class TranslateZhruFull(translate.TranslateProblem):
    """En-de translation trained on WMT corpus."""

    @property
    def additional_training_datasets(self):
        """Allow subclasses to add training datasets."""
        return []

    def source_data_files(self, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        train_datasets = _ENDE_TRAIN_DATASETS + self.additional_training_datasets
        return train_datasets if train else _ENDE_EVAL_DATASETS

    @property
    def approx_vocab_size(self):
        return 47000


@registry.register_hparams
def transformer_medium():
    hparams = transformer.transformer_base()
    hparams.hidden_size = 256
    hparams.filter_size = 1024
    hparams.batch_size = 8192
    hparams.max_length = 128
    return hparams

@registry.register_hparams
def transformer_medium_tpu():
    hparams = transformer.transformer_tpu()
    hparams.hidden_size = 256
    hparams.filter_size = 1024
    hparams.batch_size = 8192
    hparams.max_length = 128
    return hparams


@registry.register_hparams
def evolved_transformer_medium():
    hparams = evolved_transformer.evolved_transformer_base()
    hparams.hidden_size = 256
    hparams.filter_size = 1024
    hparams.batch_size = 4096
    hparams.max_length = 128
    return hparams
