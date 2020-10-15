# This script downloads the dataset locally, loads only a chunk of the dataset for the training.

import os
import tensorflow as tf
import tensorflow_datasets as tfds


os.makedirs(os.path.join("data", "raw"), exist_ok=True)
output_train = os.path.join('data', 'raw', 'train.tfrecord')
output_test = os.path.join('data', 'raw', 'test.tfrecord')


def fetch_datasets():

    train_ds, test_ds = tfds.load('fashion_mnist', split=["train", "test"])
    return train_ds, test_ds


def save_datasets(train_ds, test_ds):
    tf.data.experimental.save(train_ds, output_train)
    tf.data.experimental.save(test_ds, output_test)


train_ds, test_ds = fetch_datasets()
save_datasets(train_ds, test_ds)