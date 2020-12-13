import os
import sys
import pickle
import numpy as np
from argparse import ArgumentParser
from models.text_model_ps import TextModel
from utils.prepro_text import load_datasets, build_vocabulary, dataset_to_indices, remap_labels
from utils.data_funcs import gen_nary_ecoc, compute_ensemble_accuracy, boolean_string, get_conf_matrix

parser = ArgumentParser()
parser.add_argument("--gpu_idx", type=str, default="0", help="")
parser.add_argument("--training", type=boolean_string, default=True, help="if True, train the model")
parser.add_argument("--task", type=str, default="trec", help="[stsa or trec]")
parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--char_dim", type=int, default=50, help="char embedding dimension")
parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[2, 3, 4], help="kernel sizes for char cnn")
parser.add_argument("--filters", type=int, nargs="+", default=[25, 25, 25], help="filters for char cnn")
parser.add_argument("--emb_drop_rate", type=float, default=0.2, help="embedding drop rate")
parser.add_argument("--drop_rate", type=float, default=0.3, help="encoder drop rate")
parser.add_argument("--num_layers", type=int, default=3, help="number of layers for encoder")
parser.add_argument("--num_units", type=int, default=200, help="number of encoder units")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--epochs", type=int, default=30, help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr_decay", type=float, default=0.01, help="learning rate decay")
parser.add_argument("--grad_clip", type=float, default=5.0, help="maximal gradient value")
parser.add_argument("--num_meta_class", type=int, default=3, help="number of meta class")
parser.add_argument("--num_classifier", type=int, default=60, help="number of classifiers")
parser.add_argument("--ablation", type=boolean_string, default=False, help="")
config = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_idx

wordvec_path = os.path.join(os.path.expanduser("~"), "utilities", "embeddings", "monolingual", "cc.en.300.vec")


if config.task == "stsa":
    num_class = 5
    num_meta_class = config.num_meta_class
    num_classifier = config.num_classifier
    if num_meta_class == 2:
        name = "stsa_part_model_ecoc"
    elif num_meta_class == num_class:
        name = "stsa_part_model_ri"
    elif 2 < num_meta_class < num_class:
        name = "stsa_part_model_nary_ecoc_{}".format(num_meta_class)
    else:
        raise ValueError("num_meta_class must in [2, num_class]!!!")
    save_path = "ckpt/{}/".format(name)
    ckpt_path = save_path + "{}/"
    data_path = os.path.join("datasets", "raw", "stsa")
    files = [data_path + "/train.txt", data_path + "/dev.txt", data_path + "/test.txt"]
    pretrained_model = "ckpt/stsa_model/text_model-16"
else:
    num_class = 6
    num_meta_class = config.num_meta_class
    num_classifier = config.num_classifier
    if num_meta_class == 2:
        name = "trec_part_model_ecoc"
    elif num_meta_class == num_class:
        name = "trec_part_model_ri"
    elif 2 < num_meta_class < num_class:
        name = "trec_part_model_nary_ecoc_{}".format(num_meta_class)
    else:
        raise ValueError("num_meta_class must in [2, num_class]!!!")
    save_path = "ckpt/{}/".format(name)
    ckpt_path = save_path + "{}/"
    data_path = os.path.join("datasets", "raw", "trec")
    files = [data_path + "/train.txt", data_path + "/test.txt"]
    pretrained_model = "ckpt/trec_model/text_model-23"

# load datasets
train_data, test_data = load_datasets(files)

# build vocabulary and load pre-trained word embeddings
if not os.path.exists(os.path.join(data_path, "processed.pkl")):
    word_dict, char_dict, vectors = build_vocabulary([train_data, test_data], wordvec_path, dim=config.word_dim)
    dd = {"word_dict": word_dict, "char_dict": char_dict, "vectors": vectors}
    with open(os.path.join(data_path, "processed.pkl"), mode="wb") as handle:
        pickle.dump(dd, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(os.path.join(data_path, "processed.pkl"), mode="rb") as handle:
        dd = pickle.load(handle)
    word_dict = dd["word_dict"]
    char_dict = dd["char_dict"]
    vectors = dd["vectors"]

# convert data into indices
train_words, train_chars, train_labels = dataset_to_indices(train_data, word_dict, char_dict)
test_words, test_chars, test_labels = dataset_to_indices(test_data, word_dict, char_dict)

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
    train_words_ith, train_chars_ith, train_labels_ith = train_words.copy(), train_chars.copy(), train_labels.copy()
    test_words_ith, test_chars_ith, test_labels_ith = test_words.copy(), test_chars.copy(), test_labels.copy()
    train_labels_ith = remap_labels(train_labels_ith, ecoc_array)
    test_labels_ith = remap_labels(test_labels_ith, ecoc_array)
    model = TextModel(config, num_meta_class, word_dict, char_dict, vectors, ckpt_path=ckpt_path.format(i),
                      pretrained_model=pretrained_model)
    if config.training:
        model.train(train_words_ith, train_chars_ith, train_labels_ith, test_words_ith, test_chars_ith,
                    test_labels_ith)
    _, pred_labels = model.test(test_words_ith, test_chars_ith, test_labels_ith, batch_size=200)
    model.close_session()
    nary_ecoc_test_result.append(pred_labels)

nary_ecoc_labels = np.concatenate(nary_ecoc_test_result, axis=1)
np.savez_compressed(save_path + "pred_labels.npz", embeddings=nary_ecoc_labels)

if config.ablation:
    nl = list(range(5, num_classifier + 1, 5))
    for n in nl:
        accuracy = compute_ensemble_accuracy(nary_ecoc_labels[:, 0:n], nary_ecoc[:, 0:n], test_labels)
        print("{}: {}\t{:4.2f}%".format(n, accuracy, accuracy * 100))

accuracy = compute_ensemble_accuracy(nary_ecoc_labels, nary_ecoc, test_labels)
print(accuracy)
# get_conf_matrix(nary_ecoc_labels, nary_ecoc, test_labels, filename="assets/{}_heatmap".format(name))

with open(save_path + "results.txt", mode="w", encoding="utf-8") as f:
    f.write(str(accuracy))
