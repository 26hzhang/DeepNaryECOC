import numpy as np
import os
from keras.datasets import cifar100, cifar10
from utils.prepro_flower import load_data
from utils.prepro_text import load_datasets, get_dataset_labels
from utils.prepro_mnist import read_data_sets

(_, _), (_, cifar10_golden_labels) = cifar10.load_data()
cifar10_golden_labels = np.asarray(cifar10_golden_labels)
cifar10_golden_labels = np.reshape(cifar10_golden_labels, newshape=cifar10_golden_labels.shape[0])
np.savez_compressed("assets/results/a_golden_labels/cifar10_golden_labels.npz", embeddings=cifar10_golden_labels)

(_, _), (_, cifar100_golden_labels) = cifar100.load_data()
cifar100_golden_labels = np.asarray(cifar100_golden_labels)
cifar100_golden_labels = np.reshape(cifar100_golden_labels, newshape=cifar100_golden_labels.shape[0])
np.savez_compressed("assets/results/a_golden_labels/cifar100_golden_labels.npz", embeddings=cifar100_golden_labels)

(_, _), (_, flower_golden_labels) = load_data()
flower_golden_labels = np.asarray(flower_golden_labels)
flower_golden_labels = np.reshape(flower_golden_labels, newshape=flower_golden_labels.shape[0])
np.savez_compressed("assets/results/a_golden_labels/flower_golden_labels.npz", embeddings=flower_golden_labels)

mnist_golden_labels = read_data_sets("datasets/raw/mnist/", one_hot=False).test.labels
mnist_golden_labels = np.asarray(mnist_golden_labels)
mnist_golden_labels = np.reshape(mnist_golden_labels, newshape=mnist_golden_labels.shape[0])
np.savez_compressed("assets/results/a_golden_labels/mnist_golden_labels.npz", embeddings=mnist_golden_labels)

data_path = os.path.join("datasets", "raw", "stsa")
files = [data_path + "/train.txt", data_path + "/dev.txt", data_path + "/test.txt"]
_, test_data = load_datasets(files)
stsa_golden_labels = get_dataset_labels(test_data)
stsa_golden_labels = np.asarray(stsa_golden_labels)
stsa_golden_labels = np.reshape(stsa_golden_labels, newshape=stsa_golden_labels.shape[0])
np.savez_compressed("assets/results/a_golden_labels/stsa_golden_labels.npz", embeddings=stsa_golden_labels)

data_path = os.path.join("datasets", "raw", "trec")
files = [data_path + "/train.txt", data_path + "/test.txt"]
_, test_data = load_datasets(files)
trec_golden_labels = get_dataset_labels(test_data)
trec_golden_labels = np.asarray(trec_golden_labels)
trec_golden_labels = np.reshape(trec_golden_labels, newshape=trec_golden_labels.shape[0])
np.savez_compressed("assets/results/a_golden_labels/trec_golden_labels.npz", embeddings=trec_golden_labels)

