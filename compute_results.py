import numpy as np
import os
from utils.data_funcs import compute_ensemble_accuracy, get_conf_matrix, boolean_string, compute_q_statistics
from utils.data_funcs import compute_f1
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--directory", type=str, default="cifar10_cnn_part_nary_ecoc_4", help="")
parser.add_argument("--save_plot", type=boolean_string, default=False, help="")
config = parser.parse_args()

if "_full_" in config.directory:
    folder = os.path.join("assets", "results", "full_share", config.directory)
elif "_part_" in config.directory:
    folder = os.path.join("assets", "results", "part_share", config.directory)
else:
    folder = os.path.join("assets", "results", "private", config.directory)

label_folder = os.path.join("assets", "results", "a_golden_labels")

if "cifar10_" in config.directory:
    golden_labels = np.load(os.path.join(label_folder, "cifar10_golden_labels.npz"))["embeddings"]
elif "cifar100_" in config.directory:
    golden_labels = np.load(os.path.join(label_folder, "cifar100_golden_labels.npz"))["embeddings"]
elif "flower102_" in config.directory:
    golden_labels = np.load(os.path.join(label_folder, "flower_golden_labels.npz"))["embeddings"]
elif "mnist" in config.directory:
    golden_labels = np.load(os.path.join(label_folder, "mnist_golden_labels.npz"))["embeddings"]
elif "stsa" in config.directory:
    golden_labels = np.load(os.path.join(label_folder, "stsa_golden_labels.npz"))["embeddings"]
elif "trec" in config.directory:
    golden_labels = np.load(os.path.join(label_folder, "trec_golden_labels.npz"))["embeddings"]
else:
    raise ValueError("Unknown task!!! Only support [cifar10 | cifar100 | flower102 | mnist | stsa | trec]")

pred_labels = np.load(folder + "/pred_labels.npz")["embeddings"]
num_classifier = pred_labels.shape[1]

if num_classifier == 1:
    golden_labels = np.asarray(golden_labels)
    golden_labels = np.reshape(golden_labels, newshape=golden_labels.shape[0])
    pred_labels = np.reshape(pred_labels, newshape=pred_labels.shape[0])
    accuracy = np.mean(np.equal(pred_labels, golden_labels).astype(np.float32))
    print("Accuracy: {}\t{:4.2f}%".format(accuracy, accuracy * 100))
    if config.plot:
        get_conf_matrix(pred_labels, None, golden_labels, filename="assets/{}_heatmap".format(config.directory))
else:
    nary_ecoc = np.load(folder + "/nary_ecoc.npz")["embeddings"]
    nl = list(range(5, num_classifier + 1, 5))
    accuracies = list()
    for n in nl:
        accuracy = compute_ensemble_accuracy(pred_labels[:, 0:n], nary_ecoc[:, 0:n], golden_labels)
        accuracies.append(float("{:4.2f}".format(accuracy * 100)))
        print("Accuracy: n={}\t{}\t{:4.2f}%".format(n, accuracy, accuracy * 100))
    print(",".join([str(x) for x in nl]))
    print(",".join([str(x) for x in accuracies]))
    q_val = compute_q_statistics(pred_labels, nary_ecoc, golden_labels)
    print("Average Q-statistics: {}".format(q_val))
    micro_f1, macro_f1, weighted_f1 = compute_f1(pred_labels, nary_ecoc, golden_labels)
    print("Micro-F1 Score: {}".format(micro_f1))
    print("Macro-F1 Score: {}".format(macro_f1))
    print("Weighted Macro-F1 Score: {}".format(weighted_f1))
    get_conf_matrix(pred_labels, nary_ecoc, golden_labels, filename="assets/{}_heatmap".format(config.directory),
                    save_file=config.save_plot)
