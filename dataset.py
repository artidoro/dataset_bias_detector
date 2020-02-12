import multiprocessing
import logging
import pickle
import math
import os

import utils

class Dataset:
    """
    Class holding info about the dataset that is needed to calculate PMI.
    """
    def __init__(self, args, unigram_counter, joint_unigram_counter):
        self.args = args
        self.unigram_counter = unigram_counter
        self.joint_unigram_counter = joint_unigram_counter
        self.number_tokens = len(unigram_counter)

    def PMI_all_words(self, word):
        """
        Returns a dictionary of words in the dataset and associated pmi with the give `word`.
        The dictionary returned is sorted by pmi.
        """
        if word not in self.unigram_counter:
            logger = logging.getLogger('logger')
            logger.error('Word {} is not present in dataset or is stop word or rare word.')
            exit()
        else:
            pmi_dict = {w2:self.PMI(word, w2) for w2 in self.unigram_counter.keys()}
            pmi_dict_sorted = {word:pmi for word,pmi in sorted(pmi_dict.items(), key=lambda item: item[1], reverse=True)}
            return pmi_dict_sorted

    def PMI(self, w1, w2):
        return utils.pmi_from_counts(self.unigram_counter[w1], self.unigram_counter[w2],
            self.joint_unigram_counter[frozenset({w1, w2})], self.number_tokens)

    def save(self, path):
        with open(path, 'wb') as save_file:
            pickle.dump(vars(self), save_file)


def load_dataset(path):
    logger = logging.getLogger('logger')
    if not os.path.exists(path):
        logger.error('File path {} does not exist, cannot load.'.format(path))
    with open(path, 'rb') as load_file:
        load_attributes = pickle.load(load_file)
    return Dataset(load_attributes['args'],
        load_attributes['unigram_counter'],
        load_attributes['joint_unigram_counter'])