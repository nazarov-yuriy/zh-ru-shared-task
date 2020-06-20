from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry


@registry.register_problem
class TranslateZhruFull(translate.TranslateProblem):

    @property
    def additional_training_datasets(self):
        assert False
        return []

    def source_data_files(self, dataset_split):
        assert False
        return []

    @property
    def approx_vocab_size(self):
        return 47000  # Determines vocabulary file path
