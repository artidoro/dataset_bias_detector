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
    def __init__(self, args, single_counter, joint_counter):
        self.args = args
        self.single_counter = single_counter
        self.joint_counter = joint_counter
        self.number_tokens = sum(single_counter.values())

    def PMI_all_words(self, word):
        """
        Returns a dictionary of words in the dataset and associated pmi with the give `word`.
        The dictionary returned is sorted by pmi.
        """
        if word not in self.single_counter:
            logger = logging.getLogger('logger')
            logger.error('Word {} is not present in dataset or is stop word or rare word.')
            exit()
        else:
            pmi_dict = {w2:self.PMI(word, w2) for w2 in self.single_counter.keys()}
            pmi_dict_sorted = {word:pmi for word,pmi in sorted(pmi_dict.items(), key=lambda item: item[1], reverse=True)}
            return pmi_dict_sorted

    def PMI(self, w1, w2):
        return utils.pmi_from_counts(self.single_counter[w1], self.single_counter[w2],
            self.joint_counter[frozenset({w1, w2})], self.number_tokens)

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
        load_attributes['single_counter'],
        load_attributes['joint_counter'])