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
import pprint
import json

import dataset
import utils
import logging_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for the text classification model.')
    parser.add_argument('--unigram', default=None)
    parser.add_argument('--bigram', default=None)
    parser.add_argument('--no_premise', action='store_true')
    parser.add_argument('--no_hypothesis', action='store_true')
    parser.add_argument('--test_identity_labels', action='store_true')
    parser.add_argument('--mode', default='train', help='Whether to train, or load trained counts.')
    parser.add_argument('--save_path', default='log')
    parser.add_argument('--load_path_unigram', help='Path to saved checkpoint for unigram.')
    parser.add_argument('--load_path_bigram', help='Path to saved checkpoint for bigram.')
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

        # sentence_list = sentence_list[:2]

        # Preprocess: tokenize data and remove stop words.
        logger.info('Starting to preprocess data using {} processes.'.format(args['num_processes']))
        p = multiprocessing.Pool(args['num_processes'])
        unigram_sentence_list = p.map(utils.preprocess_sentences,
            utils.chunks(sentence_list, math.ceil(len(sentence_list)/args['num_processes'])),
            chunksize=1)
        unigram_sentence_list = list(itertools.chain.from_iterable(unigram_sentence_list))
        logger.info('Finished preprocessing.')

        # Get the bigram sentence list.
        logger.info('Starting to extract bigram lists using {} processes.'.format(args['num_processes']))
        p = multiprocessing.Pool(args['num_processes'])
        bigram_sentence_list = p.map(utils.sentence_unigram_to_bigram, unigram_sentence_list,
            chunksize=math.ceil(len(unigram_sentence_list)/args['num_processes']))
        logger.info('Finished extracting bigrams.')

        # Get unigram and bigram counters.
        single_unigram_counter, joint_unigram_counter = utils.single_and_joint_counters(args, unigram_sentence_list)
        single_bigram_counter, joint_bigram_counter = utils.single_and_joint_counters(args, bigram_sentence_list)

        # Save the dataset.
        unigram_dataset = dataset.Dataset(args, single_unigram_counter, joint_unigram_counter)
        bigram_dataset =  dataset.Dataset(args, single_bigram_counter, joint_bigram_counter)
        dt_string = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        dataset_path = os.path.join(args['save_path'], 'unigram_' + dt_string + '.dt')
        logger.info('Saving dataset: {}'.format(dataset_path))
        unigram_dataset.save(dataset_path)
        dataset_path = os.path.join(args['save_path'], 'bigram_' + dt_string + '.dt')
        logger.info('Saving dataset: {}'.format(dataset_path))
        bigram_dataset.save(dataset_path)
    else:
        # Load the dataset counts.
        logger.info('Loading dataset from checkpoint from {}'.format(args['load_path_unigram']))
        unigram_dataset = dataset.load_dataset(args['load_path_unigram'])
        logger.info('Loading dataset from checkpoint from {}'.format(args['load_path_bigram']))
        bigram_dataset = dataset.load_dataset(args['load_path_bigram'])

    if args['unigram'] is not None:
        logger.info('Calculating PMI for word {}.'.format(args['unigram']))
        pmi_dict = unigram_dataset.PMI_all_words(args['unigram'])
        logger.info('The top 5 words in terms of PMI of unigram frequency are:\n{}'.format(list(pmi_dict.items())[:20]))

    if args['bigram'] is not None:
        logger.info('Calculating PMI for word {}.'.format(args['bigram']))
        pmi_dict = bigram_dataset.PMI_all_words(args['bigram'])
        logger.info('The top 5 words in terms of PMI of bigram frequency are:\n{}'.format(list(pmi_dict.items())[:20]))

    if args['test_identity_labels']:
        logger.info('Testing all identity labels.')

        with open(args['identity_labels_path']) as identity_labels_file:
            identity_list = list(map(lambda x: x.strip(), identity_labels_file.readlines()))

        # Unigrams
        unigram_pmi_json = {}
        for unigram in tqdm.tqdm(set(identity_list).intersection(set(unigram_dataset.single_counter.keys()))):
            unigram_pmi_json[unigram] = list(unigram_dataset.PMI_all_words(unigram).items())[:30]

        unigram_file_path = os.path.join(args['save_path'], 'identity_unigram_pmi.txt')
        logger.info('Writing all pmi results for identity words to {}'.format(unigram_file_path))
        with open(unigram_file_path, 'w') as unigram_file:
            unigram_file.write(pprint.pformat(unigram_pmi_json))
        with open(unigram_file_path + 'json', 'w') as unigram_file:
            unigram_file.write(json.dumps(unigram_pmi_json))
        # Bigrams
        bigram_pmi_json = {}
        identity_bigrams = [w1 + ' ' + w2 for w1, w2 in itertools.permutations(identity_list, 2)]
        for bigram in tqdm.tqdm(set(identity_bigrams).intersection(set(bigram_dataset.single_counter.keys()))):
            bigram_pmi_json[bigram] = list(bigram_dataset.PMI_all_words(bigram).items())[:30]

        bigram_file_path = os.path.join(args['save_path'], 'identity_bigram_pmi.txt')
        logger.info('Writing all pmi results for identity words to {}'.format(bigram_file_path))
        with open(bigram_file_path, 'w') as bigram_file:
            bigram_file.write(pprint.pformat(bigram_pmi_json))
        with open(bigram_file_path + 'json', 'w') as bigram_file:
            bigram_file.write(json.dumps(bigram_pmi_json))