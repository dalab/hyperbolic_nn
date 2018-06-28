import numpy as np
import pickle
import argparse
import os
import re

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

# Path to a folder called snli_dataset/ where all binary data will be placed. 
# This folder should contain the original SNLI dataset downloaded and unzipped from https://nlp.stanford.edu/projects/snli/
parser.add_argument("--root_path", type=str, help="Root path", default='/path/to/your/snli_dataset/') 

# Original SNLI files.
parser.add_argument("--train_file", type=str, help="Train path", default='snli_1.0/snli_1.0_train.txt')
parser.add_argument("--dev_file", type=str, default='snli_1.0/snli_1.0_dev.txt')
parser.add_argument("--test_file", type=str, default='snli_1.0/snli_1.0_test.txt')

parser.add_argument("--save_train_file", type=str, default='train')
parser.add_argument("--save_dev_file", type=str, default='dev')
parser.add_argument("--save_test_file", type=str, default='test')

parser.add_argument("--save_word_to_id", type=str, default='word_to_id')
parser.add_argument("--save_id_to_word", type=str, default='id_to_word')

parser.add_argument("--num_classes", type=int, default=2)

parser.add_argument("--use_vocab", type=str2bool,
                    help="Whether to use already stored word_to_id and id_to_word dicts",
                    default="no")

args = parser.parse_args()

CLASS_TO_ID = {
    'contradiction': 2,
    'neutral': 0,
    'entailment': 1
}

if args.num_classes == 2:
    CLASS_TO_ID['contradiction'] = 0

original_filepath_train = os.path.join(args.root_path, args.train_file)
original_filepath_dev = os.path.join(args.root_path, args.dev_file)
original_filepath_test = os.path.join(args.root_path, args.test_file)

suffix = '_' + str(args.num_classes) + 'class'
save_filepath_train = os.path.join(args.root_path, args.save_train_file + suffix)
save_filepath_dev = os.path.join(args.root_path, args.save_dev_file + suffix)
save_filepath_test = os.path.join(args.root_path, args.save_test_file + suffix)

save_filepath_word2id = os.path.join(args.root_path, args.save_word_to_id)
save_filepath_id2word = os.path.join(args.root_path, args.save_id_to_word)


def get_vocab(file_paths):
    if args.use_vocab:
        word_to_id = pickle.load(open(save_filepath_word2id, 'rb'))
        id_to_word = pickle.load(open(save_filepath_id2word, 'rb'))
        print('USING SAVED VOCABULARY')
        return word_to_id, id_to_word

    word_to_id = {}
    id_to_word = []
    curr_word_id = 0

    word_to_id['<UNK>'] = 0
    id_to_word.append('<UNK>')
    curr_word_id += 1

    for file_path in file_paths:
        for line in open(file_path, 'r'):
            parts = line.strip().lower().split('\t')

            sent1 = parts[5]
            sent2 = parts[6]

            words_1 = re.split('[^a-zA-Z]', sent1)
            words_2 = re.split('[^a-zA-Z]', sent2)

            for w in words_1 + words_2:
                if w not in word_to_id:
                    word_to_id[w] = curr_word_id
                    id_to_word.append(w)
                    curr_word_id += 1

    assert len(word_to_id) == len(id_to_word)
    return word_to_id, id_to_word


def transform(file_path, word_to_id):
    transformed_data = []
    for line in open(file_path, 'r'):

        parts = line.strip().lower().split('\t')

        sent1 = parts[5]
        sent2 = parts[6]
        label = parts[0]

        if label not in CLASS_TO_ID: continue

        words_1 = re.split('[^a-zA-Z]', sent1)
        words_2 = re.split('[^a-zA-Z]', sent2)

        word_ids_1 = []
        word_ids_2 = []

        for w in words_1:
            word_ids_1.append(word_to_id[w])

        for w in words_2:
            word_ids_2.append(word_to_id[w])

        # list of words for s1 , list of words for s2 , class id
        transformed_data.append((word_ids_1, len(word_ids_1), word_ids_2, len(word_ids_2), CLASS_TO_ID[label]))

    return transformed_data


def print_stats(data):
    stats = {}
    for k in CLASS_TO_ID.keys():
        stats[CLASS_TO_ID[k]] = 0

    for _, _, _, _, label in data:
        stats[label] += 1

    for k in CLASS_TO_ID.keys():
        print(' --> ' + k + ' : ' + str(stats[CLASS_TO_ID[k]]))



def save_files():
    word_to_id, id_to_word = get_vocab([original_filepath_train, original_filepath_dev, original_filepath_test])

    train_data = transform(original_filepath_train, word_to_id)
    dev_data = transform(original_filepath_dev, word_to_id)
    test_data = transform(original_filepath_test, word_to_id)

    print('Vocab size: %d' % len(word_to_id))
    print('Training data size: %d' % len(train_data))
    print_stats(train_data)
    print('Dev data size: %d' % len(dev_data))
    print_stats(dev_data)
    print('Test data size: %d' % len(test_data))
    print_stats(test_data)

    print('Some samples: ')
    for i in range(0, 20):
        ind = np.random.randint(0, len(train_data))
        for word_id in train_data[ind][0]:
            print(id_to_word[word_id] + ' ' )
        print('')
        for word_id in train_data[ind][2]:
            print(id_to_word[word_id] + ' ')

        print(train_data[ind][4])
        print('===================')

    if not args.use_vocab:
        pickle.dump(word_to_id, open(save_filepath_word2id, 'wb'))
        pickle.dump(id_to_word, open(save_filepath_id2word, 'wb'))

    pickle.dump(train_data, open(save_filepath_train, 'wb'))
    pickle.dump(dev_data, open(save_filepath_dev, 'wb'))
    pickle.dump(test_data, open(save_filepath_test, 'wb'))


if __name__ == '__main__':
    save_files()
