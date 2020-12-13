import os
import glob
import pickle
import numpy as np
import sklearn.utils
from tqdm import tqdm
from scipy import ndimage, misc

data_path = os.path.join("datasets", "raw", "flower102")
directories = ["train", "valid", "test"]


def read_images_and_labels(directory):
    images, labels = list(), list()
    data_folder = os.path.join(data_path, directory)
    for i in tqdm(range(1, 103), total=102, desc="read images from {} directories".format(directory)):
        image_folder = os.path.join(data_folder, str(i))
        image_files = glob.glob(image_folder + "/*.jpg")
        image_files.sort()
        for image_file in image_files:
            image = ndimage.imread(image_file, mode="RGB", flatten=False)  # load image
            image = misc.imresize(image, size=(227, 227))
            images.append(image)
            labels.append(i - 1)
    images, labels = sklearn.utils.shuffle(images, labels)
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels


def load_data():
    pickle_path = os.path.join(data_path, "processed.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, mode="rb") as handle:
            data = pickle.load(handle)
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]
    else:
        x_train, y_train = read_images_and_labels("train")
        x_valid, y_valid = read_images_and_labels("valid")
        x_train = np.concatenate((x_train, x_valid), axis=0)
        y_train = np.concatenate((y_train, y_valid), axis=0)
        x_test, y_test = read_images_and_labels("test")
        data = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
        with open(pickle_path, mode="wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return (x_train, y_train), (x_test, y_test)
