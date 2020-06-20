# zh-ru-shared-task
Chinese->Russian machine translation shared task

### Data
For training model following parallel corpora were used:
* http://opus.nlpl.eu/UNPC-v1.0.php
* http://opus.nlpl.eu/TED2013-v1.1.php
* http://opus.nlpl.eu/WMT-News-v2019.php
* http://opus.nlpl.eu/OpenSubtitles-v2018.php
* https://linguatools.org/tools/corpora/wikipedia-parallel-titles-corpora/
* http://www.casmacat.eu/corpus/news-commentary.html

Additionally news and 2017 year of proza.ru from monolingual Taiga Corpus were translated to Chinese:
* https://tatianashavrina.github.io/taiga_site/

### Model
* Evolved transformer https://arxiv.org/abs/1901.11117 were chosen for model's architecture
* Tensor2Tensor https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/evolved_transformer.py were chosen for model's implementation

### Results

Train data | Public score | Private score
--- | --- | ---
Parallel only | 8.0 | ?
Parallel + back translation | 9.1 | ?
Parallel + back translation, 4x augmentation | 8.4 | ?
Parallel + back translation, 10x augmentation | 8.9 | ?
