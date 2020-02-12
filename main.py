import os
import argparse
import logging
import itertools
import collections
import multiprocessing
import math
import tqdm
import pickle
import datetime

import dataset
import utils
import logging_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for the text classification model.')
    parser.add_argument('--unigram', default=None)
    parser.add_argument('--mode', default='train', help='Whether to train, or load trained counts.')
    parser.add_argument('--save_path', default='log')
    parser.add_argument('--load_path', help='Path to saved checkpoint.')
    parser.add_argument('--data_path', default='/home/artidoro/data/snli_1.0')
    parser.add_argument('--identity_labels_path', default='identity_labels.txt')
    parser.add_argument('--num_processes', default=12, type=int)
    parser.add_argument('--min_appearance_count', default=10, type=int)
    args = vars(parser.parse_args())

    # Setup logging and create dir for checkpoints and logging.
    logger = logging_utils.setup_logging('logger', args['save_path'])

    if args['mode'] == 'train':
        # Load all sentences and filter out duplicate sentences.
        logger.info('Starting to load data.')
        sentence_list = utils.load_sentences(args)

        # Preprocess: tokenize data and remove stop words.
        logger.info('Starting to preprocess data using {} processes.'.format(args['num_processes']))
        p = multiprocessing.Pool(args['num_processes'])
        preprocessed_sentence_list = p.map(utils.preprocess_sentences,
            utils.chunks(sentence_list, math.ceil(len(sentence_list)/args['num_processes'])),
            chunksize=1)
        preprocessed_sentence_list = list(itertools.chain.from_iterable(preprocessed_sentence_list))
        logger.info('Finished preprocessing.')

        # Get counts for all words and remove low frequency words.
        logger.info('Starting to build unigram counter.')
        flattened_list = list(itertools.chain.from_iterable(preprocessed_sentence_list))
        unigram_counter = collections.Counter(flattened_list)
        logger.info('Finished building unigram counter.\nOut of {} tokens in dataset, {} are unique.'.format(
            len(flattened_list), len(unigram_counter)))
        logger.info('Removing words appearing less than 10 times.')
        low_frequency_words = {k for k,v in tqdm.tqdm(unigram_counter.items()) if v < args['min_appearance_count']}
        for word in low_frequency_words:
            del unigram_counter[word]
        logger.info('Finished removing words from counter. Unigram counter now contains {} unique words'.format(
            len(unigram_counter)))

        # Get sentence level counters.
        logger.info('Starting to build sentence level counters using {} processes.'.format(args['num_processes']))
        word_sentence_counters = list(p.starmap(utils.sentence_counter,
            [(sentence, low_frequency_words) for sentence in preprocessed_sentence_list],
            chunksize=math.ceil(len(sentence_list)/args['num_processes'])))
        logger.info('Finished building sentence level counters')

        # Get joint counts.
        logger.info('Starting to build joint occurrence counts using {} processes.'.format(args['num_processes']))
        joint_counters = p.map(utils.build_joint_counters,
            utils.chunks(word_sentence_counters, math.ceil(len(word_sentence_counters)/args['num_processes'])),
            chunksize=1)
        joint_unigram_counter = sum(joint_counters, collections.Counter())
        logger.info('Finished building joint counts. Joint unigram counter has {} unique pairs'.format(len(joint_unigram_counter)))

        # Save the dataset.
        dataset = dataset.Dataset(args, unigram_counter, joint_unigram_counter)
        dt_string = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        dataset_path = os.path.join(args['save_path'], dt_string + '.dt')
        logger.info('Saving dataset: {}'.format(dataset_path))
        dataset.save(dataset_path)
    else:
        logger.info('Loading dataset from checkpoint from {}'.format(args['load_path']))
        dataset = dataset.load_dataset(args['load_path'])

    if args['unigram'] is not None:
        logger.info('Calculating PMI for word {}.'.format(args['unigram']))
        pmi_dict = dataset.PMI_all_words(args['unigram'])
        logger.info('The top 5 words in terms of PMI of unigram frequency are:\n{}'.format(list(pmi_dict.items())[:20]))
