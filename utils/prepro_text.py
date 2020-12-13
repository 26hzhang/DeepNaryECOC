import codecs
import json
import collections
import numpy as np
from tqdm import tqdm
from collections import Counter

np.random.seed(12345)
PAD, UNK = "<PAD>", "<UNK>"
glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}

Datasets = collections.namedtuple('Datasets', ['train', 'dev', 'test'])


def write_json(filename, dataset):
    with codecs.open(filename, mode="w", encoding="utf-8") as f:
        json.dump(dataset, f)


def word_convert(word, lowercase=False, char_lowercase=False):
    if char_lowercase:
        char = [c for c in word.lower()]
    else:
        char = [c for c in word]
    if lowercase:
        word = word.lower()
    return word, char


def load_emb_vocab(wordvec_path, dim):
    vocab = list()
    with codecs.open(wordvec_path, mode="r", encoding="utf-8") as f:
        if "glove" in wordvec_path:
            total = glove_sizes[wordvec_path.split(".")[-3]]
        else:
            total = int(f.readline().lstrip().rstrip().split(" ")[0])
        for line in tqdm(f, total=total, desc="Load embedding vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2:
                continue
            if len(line) != dim + 1:
                continue
            word = line[0]
            vocab.append(word)
    return set(vocab)


def filter_emb(word_dict, wordvec_path, dim):
    vectors = np.zeros([len(word_dict), dim])
    with codecs.open(wordvec_path, mode="r", encoding="utf-8") as f:
        if "glove" in wordvec_path:
            total = glove_sizes[wordvec_path.split(".")[-3]]
        else:
            total = int(f.readline().lstrip().rstrip().split(" ")[0])
        for line in tqdm(f, total=total, desc="Load embedding vectors"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2:
                continue
            if len(line) != dim + 1:
                continue
            word = line[0]
            if word in word_dict:
                vector = [float(x) for x in line[1:]]
                word_idx = word_dict[word]
                vectors[word_idx] = np.asarray(vector)
    return vectors


def load_datasets(files):
    train, dev, test = None, None, None
    for file in files:
        dataset = list()
        with codecs.open(file, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.lstrip().rstrip()
                tokens = line.split(" ")
                words = tokens[1:]
                label = int(tokens[0])
                dataset.append((words, label))
        if "train" in file:
            train = dataset.copy()
        elif "dev" in file:
            dev = dataset.copy()
        else:
            test = dataset.copy()
    if dev is not None:
        train = train + dev
    return train, test


def build_vocabulary(datasets, wordvec_path, dim):
    word_counter, char_counter = Counter(), Counter()
    for dataset in datasets:
        for words, _ in dataset:
            for word in words:
                word, chars = word_convert(word)
                word_counter[word] += 1
                for char in chars:
                    char_counter[char] += 1
    # build word vocabulary
    emb_vocab = load_emb_vocab(wordvec_path, dim)
    word_vocab = list()
    for word, _ in word_counter.most_common():
        if word in emb_vocab:
            word_vocab.append(word)
    tmp_word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    vectors = filter_emb(tmp_word_dict, wordvec_path, dim)
    word_vocab = [PAD, UNK] + word_vocab
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    # build character vocabulary
    char_vocab = [PAD, UNK] + [char for char, count in char_counter.most_common() if count >= 20]  # threshold
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    return word_dict, char_dict, vectors


def dataset_to_indices(dataset, word_dict, char_dict):
    dataset_words, dataset_chars, dataset_labels = list(), list(), list()
    for words, label in dataset:
        words_indices, chars_indices = list(), list()
        for word in words:
            word, chars = word_convert(word)
            word_idx = word_dict[word] if word in word_dict else word_dict[UNK]
            chars_idx = [char_dict[c] if c in char_dict else char_dict[UNK] for c in chars]
            words_indices.append(word_idx)
            chars_indices.append(chars_idx)
        dataset_words.append(words_indices)
        dataset_chars.append(chars_indices)
        dataset_labels.append(label)
    return dataset_words, dataset_chars, dataset_labels


def get_dataset_labels(dataset):
    labels = []
    for _, label in dataset:
        labels.append(label)
    return labels


def dense_to_one_hot(labels_dense, num_class):
    labels_dense = np.asarray(labels_dense)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_class
    labels_one_hot = np.zeros((num_labels, num_class))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    labels_one_hot = labels_one_hot.astype(dtype=np.int32)
    return labels_one_hot.tolist()


def pad_sequences(sequences, pad_tok=None, max_length=None):
    if pad_tok is None:
        pad_tok = 0  # 0: "PAD" for words and chars, "PAD" for tags
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length


def pad_char_sequences(sequences, max_length=None, max_length_2=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
        max_length = max(map(lambda x: len(x), sequences))
    if max_length_2 is None:
        max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
        sp, sl = pad_sequences(seq, max_length=max_length_2)
        sequence_padded.append(sp)
        sequence_length.append(sl)
    sequence_padded, _ = pad_sequences(sequence_padded, pad_tok=[0] * max_length_2, max_length=max_length)
    sequence_length, _ = pad_sequences(sequence_length, max_length=max_length)
    return sequence_padded, sequence_length


def batch_iter(words, chars, labels, batch_size):
    total = len(words)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        b_words, b_seq_len = pad_sequences(words[start:end])
        b_chars, b_char_seq_len = pad_char_sequences(chars[start:end])
        # b_labels, _ = pad_sequences(labels[start:end])
        yield b_words, b_seq_len, b_chars, b_char_seq_len, labels[start:end]


def remap_labels(labels, map_array, num_meta_class=None, one_hot=False):
    labels = np.asarray(labels)
    if len(labels.shape) > 1:
        if labels.shape[1] == 1:
            labels = np.reshape(labels, newshape=labels.shape[0])
        else:
            labels = np.argmax(labels, axis=1)
    new_labels = []
    for label in labels:
        new_label = map_array[label]
        new_labels.append(new_label)
    labels = dense_to_one_hot(new_labels, num_meta_class) if one_hot else new_labels
    return labels
