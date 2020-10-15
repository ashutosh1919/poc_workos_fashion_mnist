import os
import yaml
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow import TensorSpec
import tensorflow_datasets as tfds
import sys
import numpy as np
import pickle

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

params = yaml.safe_load(open('params.yaml'))['prepare']

if len(sys.argv)!=3:
    print("Not passed enough arguements to prepare stage.")
    sys.exit(1)

print("Starting the prepare stage...")

norm_const = params['norm_const']
n_classes = params['n_classes']

os.makedirs(os.path.join("data", "prepared"), exist_ok=True)
output_set = os.path.join('data', 'prepared', 'processed.npz')

input_train = sys.argv[1]
input_test = sys.argv[2]


def read_dataset(input_path):
    ds = tf.data.experimental.load(input_path,
                                   element_spec={
                                       'image': TensorSpec(shape=(28, 28, 1), 
                                                dtype=tf.uint8, 
                                                name=None),
                                       'label': TensorSpec(shape=(), 
                                                dtype=tf.int64, name=None)})
    return ds


def unzip_data(ds):

    images = []
    labels = []
    for el in ds:
        images.append( el["image"])
        labels.append( el["label"])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def normalize_images(images):
    images = images.astype('float64')
    images /= norm_const
    return images


def convert_labels_to_categories(labels):
    return to_categorical(labels, n_classes)


def save_processed(sets, filename):
    np.savez(filename, x_train=sets[0], y_train=sets[1], x_test=sets[2], y_test=sets[3])


# Read Raw Dataset from drive

train_ds = read_dataset(input_train)
test_ds = read_dataset(input_test)


# Load slice of the data to prepare for train and evaluation
train_images, train_labels = unzip_data(list(train_ds.as_numpy_iterator())[:params["train_samples"]])
test_images, test_labels = unzip_data(list(test_ds.as_numpy_iterator())[:params["test_samples"]])

# Normalize the train and Test set images
train_images = normalize_images(train_images)
test_images = normalize_images(test_images)

train_labels = convert_labels_to_categories(train_labels)
test_labels = convert_labels_to_categories(test_labels)

processed_sets = [train_images, train_labels, test_images, test_labels]

# Save the Processed Sets to gdrive under prepare
save_processed(processed_sets, output_set)

print('Prepare stage completed successfylly...')