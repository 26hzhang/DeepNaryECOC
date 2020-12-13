import os
import numpy as np
from argparse import ArgumentParser
from models.cifar_cnn import CifarCNN
from keras.datasets import cifar100
from utils.data_funcs import gen_nary_ecoc, compute_ensemble_accuracy, remap_labels, boolean_string

parser = ArgumentParser()
parser.add_argument("--gpu_idx", type=str, default="0", help="")
parser.add_argument("--training", type=boolean_string, default=True, help="if True, train the model")
parser.add_argument("--num_meta_class", type=int, default=75, help="number of meta class")
parser.add_argument("--num_classifier", type=int, default=100, help="number of classifiers")
parser.add_argument("--ablation", type=boolean_string, default=False, help="")
config = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_idx

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

num_class = 100
num_meta_class = config.num_meta_class
num_classifier = config.num_classifier

if num_classifier == 1:  # single model
    print("build model...")
    model = CifarCNN(num_class=100, ckpt_path="ckpt/cifar100_cnn/")
    if config.training:
        model.train(x_train, y_train, x_test, y_test, batch_size=400, epochs=50)
    pred_labels = model.test(x_test, y_test)
    np.savez_compressed("ckpt/cifar100_cnn/pred_labels.npz", embeddings=pred_labels)

else:  # ensemble model
    if num_meta_class == 2:
        name = "cifar100_cnn_ecoc"
    elif num_meta_class == num_class:
        name = "cifar100_cnn_ri"
    elif 2 < num_meta_class < num_class:
        name = "cifar100_cnn_nary_ecoc_{}".format(num_meta_class)
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

    nary_ecoc_test_result = []
    for i in range(num_classifier):
        print("\nThe {}/{} classifier:\n".format(i + 1, num_classifier))
        ecoc_array = nary_ecoc[:, i]
        y_train_ = remap_labels(y_train.copy(), ecoc_array)
        y_test_ = remap_labels(y_test.copy(), ecoc_array)
        model = CifarCNN(num_class=num_meta_class, ckpt_path=ckpt_path.format(i + 1))
        if config.training:
            model.train(x_train.copy(), y_train_, x_test.copy(), y_test_, batch_size=400, epochs=50)
        pred_labels = model.test(x_test.copy(), y_test_)
        model.close_session()
        nary_ecoc_test_result.append(pred_labels)

    nary_ecoc_labels = np.concatenate(nary_ecoc_test_result, axis=1)
    np.savez_compressed(save_path + "pred_labels.npz", embeddings=nary_ecoc_labels)

    if config.ablation:
        nl = list(range(5, num_classifier + 1, 5))
        for n in nl:
            accuracy = compute_ensemble_accuracy(nary_ecoc_labels[:, 0:n], nary_ecoc[:, 0:n], y_test)
            print("{}: {}\t{:4.2f}%".format(n, accuracy, accuracy * 100))

    accuracy = compute_ensemble_accuracy(nary_ecoc_labels, nary_ecoc, y_test)
    print(accuracy)

    with open(save_path + "results.txt", mode="w", encoding="utf-8") as f:
        f.write(str(accuracy))
