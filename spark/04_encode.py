from tensor2tensor.data_generators import generator_utils
import youtokentome as yttm

MAX_LEN = 128
bpe = yttm.BPE(model="models/model_ruzh_47k.yttm")


def gen(path):
    dfs = spark.read.load(path)
    for row in dfs.head(1000000000):
        sample = {
            "inputs": bpe.encode([row.zh], eos=True)[0][:MAX_LEN],
            "targets": bpe.encode([row.ru], eos=True)[0][:MAX_LEN]
        }
        yield sample


total = 32
for i in range(0, total):
    generator_utils.generate_files(
        gen("hdfs://ryzen:9000/user/root/dataset/mt/shuffled-ru-zh.parquet/part-%05d-*" % i),
        ["tfrecords/translate_zhru-train-%05d-of-%05d-unshuffled" % (i, total)]
    )

all_paths = ["tfrecords/translate_zhru-train-%05d-of-%05d-unshuffled" % (i, total) for i in range(total)]
generator_utils.shuffle_dataset(all_paths)
