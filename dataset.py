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
    def __init__(self, args, unigram_counter, word_sentence_counters):
        self.args = args
        self.unigram_counter = unigram_counter
        self.word_sentence_counters = word_sentence_counters
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
            p = multiprocessing.Pool(self.args['num_processes'])
            dataset_words = self.unigram_counter.keys()
            pmi_list =  list(p.starmap(self.PMI, [(word, dataset_word) for dataset_word in dataset_words],
                chunksize=math.ceil(len(dataset_words)/self.args['num_processes'])))
            pmi_dict = {word:pmi for word,pmi in zip(dataset_words, pmi_list)}
            pmi_dict_sorted = {word:pmi for word,pmi in sorted(pmi_dict.items(), key=lambda item: item[1], reverse=True)}
            return pmi_dict_sorted

    def PMI(self, w1, w2):
        joint_count = sum([min(sentence_counter[w1], sentence_counter[w2]) for sentence_counter in self.word_sentence_counters])
        return utils.pmi_from_counts(self.unigram_counter[w1], self.unigram_counter[w2], joint_count, self.number_tokens)

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
        load_attributes['word_sentence_counters'])