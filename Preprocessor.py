import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle
import matplotlib.image as mpimg


def load_model(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print('-Reading {} images:'.format(train_path.replace('/', '')))
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            print(image.shape)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase.split('.')[-3])
            cls.append(fld)
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)


    return images, labels, ids, cls


class DataSet(object):
  def __init__(self, images, labels, ids, cls):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""
    self._num_examples = images.shape[0]
    print('\nNum example: {}\n'.format(self._num_examples))

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    # Convert from [0, 255] -> [0.0, 1.0].

    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._ids = ids
    self._cls = cls
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def ids(self):
    return self._ids

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # # Shuffle the data (maybe)
      # perm = np.arange(self._num_examples)
      # np.random.shuffle(perm)
      # self._images = self._images[perm]
      # self._labels = self._labels[perm]
      # Start next epoch

      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples 
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_model_sets(train_path, image_size, classes):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, ids, cls = load_model(train_path, image_size, classes)
  images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

  data_sets.train = DataSet(images, labels, ids, cls)
  return data_sets

def read_test_sets(train_path, image_size, classes):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, ids, cls = load_model(train_path, image_size, classes)
  # images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

  data_sets.train = DataSet(images, labels, ids, cls)
  return data_sets
