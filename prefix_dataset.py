import numpy as np
import pickle
import argparse
import os
import re
import errno
from random import randint
import random


def str2bool(answer):
    answer = answer.lower()
    if answer in ['y', 'yes']:
        return True

    if answer in ['n', 'no']:
        return False

    print('Invalid answer: ' + answer)
    print('Exiting..')
    exit()


parser = argparse.ArgumentParser()

parser.add_argument("--root_path", type=str, help="Root path", default='/path/to/a/folder/where/prefix/datasets/will/be/created/')

parser.add_argument("--vocab_size", type=int, default=100)
parser.add_argument("--max_sent_len", type=int, default=20)
parser.add_argument("--percent_noise", type=int, default=10)

args = parser.parse_args()

percent_noise = args.percent_noise
vocab_size = args.vocab_size
max_sent_len = args.max_sent_len

train_size = 500000
dev_size = 10000
test_size = 10000

dataset_name = 'prefix_' + str(percent_noise) + '_dataset'
dir_name = os.path.join(args.root_path, dataset_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

save_filepath_train = os.path.join(args.root_path, dataset_name + '/train')
save_filepath_dev = os.path.join(args.root_path, dataset_name + '/dev')
save_filepath_test = os.path.join(args.root_path, dataset_name + '/test')

save_filepath_word2id = os.path.join(args.root_path, dataset_name + '/word_to_id')
save_filepath_id2word = os.path.join(args.root_path, dataset_name + '/id_to_word')


def get_vocab():
    word_to_id = {}
    id_to_word = []
    curr_word_id = 0

    word_to_id['<UNK>'] = 0
    id_to_word.append('<UNK>')
    curr_word_id += 1

    for w in range(vocab_size):
        word = 'W'+ str(w)
        word_to_id[word] = curr_word_id
        id_to_word.append(word)
        curr_word_id += 1

    assert len(word_to_id) == len(id_to_word)
    return word_to_id, id_to_word


def gen_data(data_size):
    data = []
    for _ in range(int(data_size / 2)):
        sent2_len = randint(1, max_sent_len) # Shorter sentence
        sent1_len = sent2_len + randint(0, max_sent_len - sent2_len)

        random_sent2 = [randint(1, vocab_size) for _ in range(sent2_len)]
        sent1_prefix = [randint(1, vocab_size) for _ in range(sent2_len)] # Shorter sentence
        sent1 = sent1_prefix + [randint(1, vocab_size) for _ in range(sent1_len - sent2_len)]

        # Randomly change percent_noise % of words in the sentence to create a noisy version.
        noisy_sent1_prefix = list(sent1_prefix)
        indices = list(range(sent2_len))
        random.shuffle(indices)
        for i in range(int(percent_noise / 100.0 * sent2_len)):
            noisy_sent1_prefix[indices[i]] = randint(1, vocab_size)

        for label in [0, 1]:
            if label == 0:
                # list of words for s1 , list of words for s2 , class id
                data.append((sent1, len(sent1), random_sent2, len(random_sent2), label))
            else:
                # list of words for s1 , list of words for s2 , class id
                data.append((sent1, len(sent1), noisy_sent1_prefix, len(noisy_sent1_prefix), label))

    return data


def print_stats(data):
    stats = {}
    for k in [0, 1]:
        stats[k] = 0

    for _, _, _, _, label in data:
        stats[label] += 1

    for k in [0, 1]:
        print(' --> ' + str(k) + ' : ' + str(stats[k]))



def save_files():
    word_to_id, id_to_word = get_vocab()

    train_data = gen_data(train_size)
    dev_data = gen_data(dev_size)
    test_data = gen_data(test_size)

    print('Vocab size: %d' % len(word_to_id))
    print('Training data size: %d' % len(train_data))
    print_stats(train_data)
    print('Dev data size: %d' % len(dev_data))
    print_stats(dev_data)
    print('Test data size: %d' % len(test_data))
    print_stats(test_data)

    print('Some samples: ')
    for i in range(0, 20):
        ind = i
        print('s1 = ' + ' '.join(id_to_word[word_id] for word_id in train_data[ind][0]))
        print('s2 = ' + ' '.join(id_to_word[word_id] for word_id in train_data[ind][2]))
        print('label = ' + str(train_data[ind][4]))
        print('===================')

    pickle.dump(word_to_id, open(save_filepath_word2id, 'wb'))
    pickle.dump(id_to_word, open(save_filepath_id2word, 'wb'))

    pickle.dump(train_data, open(save_filepath_train, 'wb'))
    pickle.dump(dev_data, open(save_filepath_dev, 'wb'))
    pickle.dump(test_data, open(save_filepath_test, 'wb'))


if __name__ == '__main__':
    save_files()
