import os
import logging
import spacy
import json
import multiprocessing
import math
import tqdm
import collections

def load_sentences(args):
    """
    Loads sentences from the jsonl files in the repo.
    Extracts `sentence1` and `sentence2` fields from the jsonl.
    Removes duplicate sentences.
    """
    logger = logging.getLogger('logger')
    sentence_list = []
    for file_name in os.listdir(args['data_path']):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(args['data_path'], file_name)
            logger.info('Loading sentences from file {}.'.format(file_path))
            with open(file_path) as data_file:
                data_lines = data_file.readlines()
                data_json = list(map(lambda x: json.loads(x), data_lines))
                sentence_list += list(map(lambda x: x['sentence1'], data_json))
                sentence_list += list(map(lambda x: x['sentence2'], data_json))

    logger.info('Starting to remove duplicates from {} sentences loaded.'.format(len(sentence_list)))
    unique_sent_list = list(set(sentence_list))
    logger.info('Done removing duplicates. Loaded {} unique sentences.'.format(len(unique_sent_list)))
    return unique_sent_list

def preprocess_sentences(sentence_list):
    """
    Preprocesses sentences with the following transformations:
        1. lowercases
        2. tokenizes
        3. removes stop words
    """
    # TODO: with open(args['identity_labels_path']) as identity_labels_file:
    with open('identity_labels.txt') as identity_labels_file:
        identity_list = list(map(lambda x: x.strip(), identity_labels_file.readlines()))
    # Remove from the stop words list the ones that appear in the identity list.
    nlp = spacy.load("en_core_web_sm")
    nlp.Defaults.stop_words -= set(identity_list)

    def preprocess_sentence(sentence):
        tokenized_sentence = nlp(sentence.lower(), disable=['parser', 'tagger', 'ner'])
        return [token.text for token in tokenized_sentence if not token.is_stop]

    preprocessed_sentence_list = [preprocess_sentence(sentence) for sentence in tqdm.tqdm(sentence_list)]
    return preprocessed_sentence_list

def sentence_counter(sentence, low_frequency):
    """
    Returns the counter of words in the sentence that are not in `low_frequency`.
    `sentence` should be a list of words.
    """
    return collections.Counter([word for word in sentence if word not in low_frequency])

def chunks(l, n):
    """
    Chunks a list in parts of size `n`.
    This is used for multiprocessing the various chunks.
    """
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def pmi_from_counts(c1, c2, c12, n):
    if c12 == 0:
        return float('-inf')
    return math.log(n * c12 / (c1 * c2), 2)
