import numpy as np
from scipy.spatial.distance import hamming
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


def boolean_string(bool_str):
    bool_str = bool_str.lower()
    if bool_str not in {"false", "true"}:
        raise ValueError("Not a valid boolean string!!!")
    return bool_str == "true"


def remap_labels(labels, map_array):
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
    new_labels = np.asarray(new_labels)
    return new_labels


def check_gen_array(gen_array, arrays):
    for array in arrays:
        if np.array_equal(gen_array, array):
            return True
    return False


def gen_nary_ecoc(num_class, num_meta_class, num_classifier):
    # according to the paper: 30 <= num_classifier < 10*log2(num_class), 3 <= num_meta_class < num_class
    if num_meta_class < num_class:
        nary_ecoc = []
        for _ in range(num_classifier):
            meta_classes = np.random.randint(num_meta_class, size=(num_class, 1))
            while check_gen_array(meta_classes, nary_ecoc) and num_meta_class > 2:
                meta_classes = np.random.randint(num_meta_class, size=(num_class, 1))
            nary_ecoc.append(meta_classes)
        nary_ecoc = np.concatenate(nary_ecoc, axis=1)
    else:
        meta_classes = [x for x in range(num_class)]
        nary_ecoc = np.asarray([meta_classes for _ in range(num_classifier)])
        nary_ecoc = np.transpose(nary_ecoc)
    return nary_ecoc


def compute_ensemble_label(array, meta_class_labels):
    result = []
    for i in range(meta_class_labels.shape[0]):
        hamming_distance = hamming(array, meta_class_labels[i, :])
        result.append(hamming_distance)
    label = np.argmin(np.asarray(result), axis=0)
    return label


def convert_predict_labels(predictions, meta_class_labels):
    pred_labels = []
    for i in range(predictions.shape[0]):
        label = compute_ensemble_label(predictions[i], meta_class_labels)
        pred_labels.append(label)
    pred_labels = np.asarray(pred_labels)
    return pred_labels


def compute_ensemble_accuracy(predictions, meta_class_labels, golden_labels):
    golden_labels = np.asarray(golden_labels)
    pred_labels = convert_predict_labels(predictions, meta_class_labels)
    golden_labels = np.reshape(golden_labels, newshape=golden_labels.shape[0])
    return np.mean(np.equal(pred_labels, golden_labels))


def compute_f1(predictions, meta_class_labels, golden_labels):
    golden_labels = np.asarray(golden_labels)
    golden_labels = np.reshape(golden_labels, newshape=golden_labels.shape[0])
    pred_labels = convert_predict_labels(predictions, meta_class_labels)
    micro_f1 = f1_score(golden_labels, pred_labels, average="micro")
    macro_f1 = f1_score(golden_labels, pred_labels, average="macro")
    weighted_f1 = f1_score(golden_labels, pred_labels, average="weighted")
    return micro_f1, np.mean(macro_f1), np.mean(weighted_f1)


def compute_q_statistics(predictions, meta_class_labels, golden_labels):
    num_classifier = predictions.shape[1]
    num_examples = predictions.shape[0]
    q_val = 0.0
    for i in range(num_classifier - 1):
        for j in range(i, num_classifier):
            pred_i = predictions[:, i]
            meta_i = remap_labels(golden_labels.copy(), meta_class_labels[:, i])
            pred_j = predictions[:, j]
            meta_j = remap_labels(golden_labels.copy(), meta_class_labels[:, j])
            n11, n00, n01, n10 = 0, 0, 0, 0
            for idx in range(num_examples):
                if pred_i[idx] == meta_i[idx] and pred_j[idx] == meta_j[idx]:
                    n11 += 1
                if pred_i[idx] != meta_i[idx] and pred_j[idx] != meta_j[idx]:
                    n00 += 1
                if pred_i[idx] == meta_i[idx] and pred_j[idx] != meta_j[idx]:
                    n01 += 1
                if pred_i[idx] != meta_i[idx] and pred_j[idx] == meta_j[idx]:
                    n10 += 1
            if n11 * n00 + n01 * n10 == 0:
                q = 0
            else:
                q = float(n11 * n00 - n01 * n10) / float(n11 * n00 + n01 * n10)
            q_val += q
    q_val = 2 * q_val / float(num_classifier * (num_classifier - 1))
    return q_val


def get_conf_matrix(predictions, meta_class_labels, golden_labels, filename=None, save_file=False):
    if meta_class_labels is not None:
        pred_labels = convert_predict_labels(predictions, meta_class_labels)
    else:
        pred_labels = np.reshape(predictions, newshape=predictions.shape[0])
    golden_labels = np.asarray(golden_labels)
    golden_labels = np.reshape(golden_labels, newshape=golden_labels.shape[0])
    cm = confusion_matrix(y_true=golden_labels, y_pred=pred_labels)
    cm = cm.astype(dtype=np.float32) / cm.sum(axis=1)[:, np.newaxis] * 100.0
    # plot
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.05, right=1.0, bottom=0.05, top=1.0)
    plt.style.use('classic')
    sns.set(font_scale=2.0)
    sns.heatmap(cm, annot=True, cmap="Reds", fmt=".1f", square=True, cbar_kws={"shrink": 0.9},
                annot_kws={"size": 30 if cm.shape[0] < 10 else 26, 'fontname': "Times New Roman"})
    plt.rcParams["font.family"] = "Times New Roman"
    if save_file:
        plt.savefig('{}.pdf'.format(filename), format="pdf")
    plt.show()
