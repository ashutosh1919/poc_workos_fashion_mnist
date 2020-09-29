# This script downloads the dataset locally, loads only a chunk of the dataset for the training.

import os
import yaml
import tensorflow as tf
import tensorflow_datasets as tfds
import sys

params = yaml.safe_load(open('params.yaml'))['extract']

train_samples = params['train_samples']
test_samples = params['test_samples']

os.makedirs(os.path.join("data", "raw"), exist_ok=True)
output_train = os.path.join('data', 'raw', 'train.tfrecord')
output_test = os.path.join('data', 'raw', 'test.tfrecord')

def fetch_datasets(train_samples=10000, test_samples=100):
    train_ds = tfds.load('fashion_mnist', split=f'train[:{train_samples}]')
    test_ds = tfds.load('fashion_mnist', split=f'test[:{test_samples}]')
    return train_ds, test_ds

def count_elements(ds, ds_type):
    ct = 0
    for x in ds:
        ct += 1
    print(f'{ds_type} data containes {ct} data points.')

def save_datasets(train_ds, test_ds):
    tf.data.experimental.save(train_ds, output_train)
    tf.data.experimental.save(test_ds, output_test)
    count_elements(train_ds, "Training")
    count_elements(test_ds, "Testing")

train_ds, test_ds = fetch_datasets(train_samples, test_samples)
save_datasets(train_ds, test_ds)