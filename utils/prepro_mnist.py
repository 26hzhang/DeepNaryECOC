import os
import time
import gzip
import numpy
import random
import collections
from six.moves import urllib
from tensorflow.python.framework import dtypes, random_seed
from tensorflow.python.platform import gfile

# codes borrowed from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/base.py
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

_RETRIABLE_ERRNOS = {
    110,  # Connection timed out [socket.py]
}

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def retry(initial_delay, max_delay, factor=2.0, jitter=0.25, is_retriable=None):
    return _internal_retry(initial_delay=initial_delay, max_delay=max_delay, factor=factor, jitter=jitter,
                           is_retriable=is_retriable)


def _internal_retry(initial_delay, max_delay, factor=2.0, jitter=0.25, is_retriable=None):
    if factor < 1:
        raise ValueError('factor must be >= 1; was %f' % (factor,))
    if jitter >= 1:
        raise ValueError('jitter must be < 1; was %f' % (jitter,))

    def delays():
        delay = initial_delay
        while delay <= max_delay:
            yield delay * random.uniform(1 - jitter, 1 + jitter)
            delay *= factor

    def wrap(fn):
        def wrapped_fn(*args, **kwargs):
            for delay in delays():
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if is_retriable is None:
                        continue
                    if is_retriable(e):
                        time.sleep(delay)
                    else:
                        raise
            return fn(*args, **kwargs)
        return wrapped_fn

    return wrap


def _is_retriable(e):
    return isinstance(e, IOError) and e.errno in _RETRIABLE_ERRNOS


@_internal_retry(initial_delay=1.0, max_delay=16.0, is_retriable=_is_retriable)
def urlretrieve_with_retry(url, filename=None):
    return urllib.request.urlretrieve(url, filename)


def maybe_download(filename, work_directory, source_url):
    if not gfile.Exists(work_directory):
        gfile.MakeDirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not gfile.Exists(filepath):
        temp_file_name, _ = urlretrieve_with_retry(source_url)
        gfile.Copy(temp_file_name, filepath)
        with gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
        return labels


class DataSet(object):
    def __init__(self, images, labels, fake_data=False, one_hot=False, dtype=dtypes.float32, reshape=True, seed=None):
        seed1, seed2 = random_seed.get_seed(seed)
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape,
                                                                                               labels.shape))
            self._num_examples = images.shape[0]
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                images = images.astype(numpy.float32)
                images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def remap_labels(self, map_array, num_meta_class=None, one_hot=False):
        labels = self._labels
        if len(labels.shape) > 1:
            if labels.shape[1] == 1:
                labels = numpy.reshape(labels, newshape=labels.shape[0])
            else:
                labels = numpy.argmax(labels, axis=1)
        new_labels = []
        for label in labels:
            new_label = map_array[label]
            new_labels.append(new_label)
        self._labels = numpy.asarray(new_labels)
        if one_hot:
            self._labels = dense_to_one_hot(self._labels, num_meta_class)

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_label for _ in range(batch_size)]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate((
                labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=dtypes.float32, reshape=True, validation_size=5000,
                   seed=None, source_url=None):
    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)
        train = fake()
        validation = fake()
        test = fake()
        return Datasets(train=train, validation=validation, test=test)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not source_url:  # empty string check
        source_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"  # http://yann.lecun.com/exdb/mnist/
    train_images_name = 'train-images-idx3-ubyte.gz'
    train_labels_name = 'train-labels-idx1-ubyte.gz'
    test_images_name = 't10k-images-idx3-ubyte.gz'
    test_labels_name = 't10k-labels-idx1-ubyte.gz'
    local_file = maybe_download(train_images_name, train_dir, source_url + train_images_name)
    with gfile.Open(local_file, 'rb') as f:
        train_images = extract_images(f)
    local_file = maybe_download(train_labels_name, train_dir, source_url + train_labels_name)
    with gfile.Open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)
    local_file = maybe_download(test_images_name, train_dir, source_url + test_images_name)
    with gfile.Open(local_file, 'rb') as f:
        test_images = extract_images(f)
    local_file = maybe_download(test_labels_name, train_dir, source_url + test_labels_name)
    with gfile.Open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)
    if not 0 <= validation_size <= len(train_images):
        raise ValueError('Validation size should be between 0 and {}. Received: {}.'.format(len(train_images),
                                                                                            validation_size))
    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]
    options = dict(dtype=dtype, reshape=reshape, seed=seed)
    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)
    return Datasets(train=train, validation=validation, test=test)


def load_mnist(train_dir=None):
    return read_data_sets(train_dir)
