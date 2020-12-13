import os
import sys
import numpy as np
from argparse import ArgumentParser
from utils.prepro_mnist import read_data_sets
from models.mnist_lenet import MnistLeNet
from utils.data_funcs import gen_nary_ecoc, compute_ensemble_accuracy, boolean_string, get_conf_matrix

parser = ArgumentParser()
parser.add_argument("--gpu_idx", type=str, default="0", help="")
parser.add_argument("--training", type=boolean_string, default=True, help="if True, train the model")
parser.add_argument("--num_meta_class", type=int, default=3, help="number of meta class")
parser.add_argument("--num_classifier", type=int, default=60, help="number of classifiers")
parser.add_argument("--ablation", type=boolean_string, default=False, help="")
config = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_idx

num_class = 10
num_meta_class = config.num_meta_class
num_classifier = config.num_classifier

if num_classifier == 1:  # single model
    mnist = read_data_sets("datasets/raw/mnist/", one_hot=True)
    train_dataset, test_dataset = mnist.train, mnist.test
    model = MnistLeNet(num_class=10, ckpt_path="ckpt/mnist_lenet/")
    if config.training:
        model.train(train_dataset, test_dataset)
    pred_labels = model.test(test_dataset)
    np.savez_compressed("ckpt/mnist_lenet/pred_labels.npz", embeddings=pred_labels)
    get_conf_matrix(pred_labels, None, np.argmax(mnist.test.labels, axis=1), filename="assets/mnist_lenet_sm_heatmap")

else:  # ensemble model
    if num_meta_class == 2:
        name = "mnist_lenet_ecoc"
    elif num_meta_class == num_class:
        name = "mnist_lenet_ri"
    elif 2 < num_meta_class < num_class:
        name = "mnist_lenet_nary_ecoc_{}".format(num_meta_class)
    else:
        raise ValueError("num_meta_class must in [2, num_class]!!!")
    save_path = "ckpt/{}/".format(name)
    ckpt_path = save_path + "{}/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if config.training:
        nary_ecoc = gen_nary_ecoc(num_class=num_class, num_meta_class=num_meta_class, num_classifier=num_classifier)
        np.savez_compressed(save_path + "nary_ecoc.npz", embeddings=nary_ecoc)
    else:
        nary_ecoc = np.load(save_path + "nary_ecoc.npz")["embeddings"]

    # start training...
    nary_ecoc_test_result = []
    for i in range(num_classifier):
        sys.stdout.write("\nThe {}/{} classifier:\n".format(i + 1, num_classifier))
        sys.stdout.flush()
        ecoc_array = nary_ecoc[:, i]
        mnist = read_data_sets("datasets/raw/mnist/", one_hot=False)
        train_dataset, test_dataset = mnist.train, mnist.test
        train_dataset.remap_labels(ecoc_array, num_meta_class, one_hot=True)
        test_dataset.remap_labels(ecoc_array, num_meta_class, one_hot=True)
        model = MnistLeNet(num_class=num_meta_class, ckpt_path=ckpt_path.format(i + 1))
        if config.training:
            model.train(train_dataset, test_dataset)
        model.restore_last_session()
        pred_labels = model.test(test_dataset)
        model.close_session()
        nary_ecoc_test_result.append(pred_labels)

    nary_ecoc_labels = np.concatenate(nary_ecoc_test_result, axis=1)
    np.savez_compressed(save_path + "pred_labels.npz", embeddings=nary_ecoc_labels)

    mnist = read_data_sets("datasets/raw/mnist/", one_hot=False)
    test_dataset = mnist.test
    golden_labels = test_dataset.labels

    if config.ablation:
        nl = list(range(5, num_classifier + 1, 5))
        for n in nl:
            accuracy = compute_ensemble_accuracy(nary_ecoc_labels[:, 0:n], nary_ecoc[:, 0:n], golden_labels)
            print("{}: {}\t{:4.2f}%".format(n, accuracy, accuracy * 100))

    accuracy = compute_ensemble_accuracy(nary_ecoc_labels, nary_ecoc, golden_labels)
    print(accuracy)
    get_conf_matrix(nary_ecoc_labels, nary_ecoc, golden_labels, filename="assets/{}_heatmap".format(name))

    with open(save_path + "results.txt", mode="w", encoding="utf-8") as f:
        f.write(str(accuracy))
