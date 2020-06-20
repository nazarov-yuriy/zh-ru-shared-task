import gc
from multiprocessing import Pool
import os
import pickle
import tensorflow.compat.v1 as tf
from tensor2tensor.data_generators import generator_utils

SHARDS = 16


def gen(root_path, shard, filters=None):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if '.pickle.' in file:
                file_path = root + '/' + file
                if hash(file_path) % SHARDS == shard and (not filters or [x for x in filters if x + '/' in file_path]):
                    with open(file_path, 'rb') as f:
                        for el in pickle.load(f):
                            yield el
                        gc.collect()


def make_tfrecords(arg):
    shard, all_paths = arg
    generator_utils.generate_files(gen("../data/augmentation", shard), all_paths)


def main():
    total = 256
    assert 0 == total % SHARDS
    all_paths = [
        '../data/augmentation_tfrecords_all/translate_zhru_my-train-%05d-of-%05d-unshuffled' % (i, total)
        for i in range(total)
    ]
    p = Pool(SHARDS)
    print(p.map(make_tfrecords, [
        (i, all_paths[i::SHARDS]) for i in range(total // SHARDS)
    ]))
    generator_utils.shuffle_dataset(all_paths)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
