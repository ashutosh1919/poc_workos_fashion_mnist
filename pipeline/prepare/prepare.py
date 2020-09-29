import os
import yaml
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow import TensorSpec
import tensorflow_datasets as tfds
import sys
import numpy as np
import pickle

params = yaml.safe_load(open('params.yaml'))['prepare']

if len(sys.argv)!=3:
    print("Not passed enough arguements to prepare stage.")
    sys.exit(1)

print("Starting the prepare stage...")

norm_const = params['norm_const']
n_classes = params['n_classes']

os.makedirs(os.path.join("data", "prepared"), exist_ok=True)
output_train = os.path.join('data', 'prepared', 'train.pkl')
output_test = os.path.join('data', 'prepared', 'test.pkl')

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
    for el in train_ds:
        images.append( el["image"].numpy())
        labels.append( el["label"].numpy())
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def normalize_images(images):
    images = images.astype('float64')
    images /= norm_const
    return images

def convert_labels_to_categories(labels):
    return to_categorical(labels, n_classes)

def save_as_pkl(images, labels, filename):
    data = {"images": images, "labels": labels}
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

train_ds = read_dataset(input_train)
test_ds = read_dataset(input_test)

train_images, train_labels = unzip_data(train_ds)
test_images, test_labels = unzip_data(test_ds)

train_images = normalize_images(train_images)
test_images = normalize_images(test_images)

train_labels = convert_labels_to_categories(train_labels)
test_labels = convert_labels_to_categories(test_labels)

save_as_pkl(train_images, train_labels, output_train)
save_as_pkl(test_images, test_labels, output_test)

print('Prepare stage completed successfylly...')