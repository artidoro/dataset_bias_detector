import os
import logging
import spacy
import json
import multiprocessing
import math
import tqdm
import collections
import itertools

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
                if not args['no_premise']:
                    sentence_list += list(map(lambda x: x['sentence1'], data_json))
                if not args['no_hypothesis']:
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
        4. remove punctuation
    """
    # TODO: with open(args['identity_labels_path']) as identity_labels_file:
    with open('identity_labels.txt') as identity_labels_file:
        identity_list = list(map(lambda x: x.strip(), identity_labels_file.readlines()))
    # Remove from the stop words list the ones that appear in the identity list.
    nlp = spacy.load("en_core_web_sm")
    nlp.Defaults.stop_words -= set(identity_list)

    def preprocess_sentence(sentence):
        tokenized_sentence = nlp(sentence.lower(), disable=['parser', 'tagger', 'ner'])
        return [token.text for token in tokenized_sentence if not (token.is_stop or token.is_punct)]

    preprocessed_sentence_list = [preprocess_sentence(sentence) for sentence in tqdm.tqdm(sentence_list)]
    return preprocessed_sentence_list

def sentence_counter(sentence, low_frequency):
    """
    Returns the counter of words in the sentence that are not in `low_frequency`.
    `sentence` should be a list of words.
    """
    return collections.Counter([word for word in sentence if word not in low_frequency])

def sentence_unigram_to_bigram(sentence):
    return [w1 + ' ' + w2 for w1,w2 in zip(sentence[:-1], sentence[1:])]

def single_and_joint_counters(args, sentence_list):
    logger = logging.getLogger('logger')
    # Get counts for all words and remove low frequency words.
    logger.info('Starting to build unigram counter.')
    flattened_list = list(itertools.chain.from_iterable(sentence_list))
    single_counter = collections.Counter(flattened_list)
    logger.info('Finished building unigram counter.\nOut of {} tokens in dataset, {} are unique.'.format(
        len(flattened_list), len(single_counter)))
    logger.info('Removing words appearing less than 10 times.')
    low_frequency_words = {k for k,v in tqdm.tqdm(single_counter.items()) if v < args['min_appearance_count']}
    for word in low_frequency_words:
        del single_counter[word]
    logger.info('Finished removing words from counter. Unigram counter now contains {} unique words'.format(
        len(single_counter)))

    # Get sentence level counters.
    logger.info('Starting to build sentence level counters using {} processes.'.format(args['num_processes']))
    p = multiprocessing.Pool(args['num_processes'])
    word_sentence_counters = list(p.starmap(sentence_counter,
        [(sentence, low_frequency_words) for sentence in sentence_list],
        chunksize=math.ceil(len(sentence_list)/args['num_processes'])))
    logger.info('Finished building sentence level counters')

    # Get joint counts.
    logger.info('Starting to build joint occurrence counts using {} processes.'.format(args['num_processes']))
    p = multiprocessing.Pool(args['num_processes'])
    joint_counters = p.map(build_joint_counters,
        chunks(word_sentence_counters, math.ceil(len(word_sentence_counters)/args['num_processes'])),
        chunksize=1)
    joint_counter = sum(joint_counters, collections.Counter())
    logger.info('Finished building joint counts. Joint unigram counter has {} unique pairs'.format(len(joint_counters)))
    return single_counter, joint_counter

def build_joint_counters(sentence_counters):
    """
    Builds counter of joint occurrences. The key of the counter is a frozenset of two words
    so that occurrence order does not matter.
    """
    joint_counter = collections.Counter()
    for counter in tqdm.tqdm(sentence_counters):
        for w1, w2 in itertools.combinations(counter.keys(), 2):
            joint_counter[frozenset({w1, w2})] += min(counter[w1], counter[w2])
    return joint_counter

def chunks(l, n):
    """
    Chunks a list in parts of size `n`.
    This is used for multiprocessing the various chunks.
    """
    ret_list = []
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        ret_list.append(l[i:i+n])
    return ret_list

def pmi_from_counts(c1, c2, c12, n):
    if c12 == 0:
        return float('-inf')
    return math.log(n * c12 / (c1 * c2), 2)
