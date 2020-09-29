"""
Preprocess script has the code to split the fashion mnist dataset into train and val. After that applying normalization.
"""

import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from pipeline.config import Config
from datetime import datetime


def load_data():
    """
    Load a chunk of the npy format saved dataset in the memory for training and evaluation.

    args->
    train_size: number of data points to be used for training, default - 60000)
    val_size: number of data points to be used for validation, default - 10000)

    returns : numpy arrays for x_train, y_train, x_test, y_test
    """
    sets = [np.load(os.path.join(Config.DATA_DIR, "raw_data", f"{key}.npy")) for key in Config.KEYS]

    return sets[0][0:Config.TRAIN_SIZE], sets[1][0:Config.TRAIN_SIZE], sets[2][0:Config.TEST_SIZE], \
           sets[3][0:Config.TEST_SIZE]


def split_dataset(sets):
    x_train, y_train, x_test, y_test = sets

    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, Config.NUM_CLASSES)
    y_test = to_categorical(y_test, Config.NUM_CLASSES)

    return x_train, y_train, x_test, y_test


def normalization(sets):
    x_train, y_train, x_test, y_test = sets
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test


def process_data():
    # Fetches the dataset from the source

    # Load chunk of the dataset for training, default loads the whole dataset
    print("Load raw dataset chunk for processing ...")
    chunk = load_data()
    sets = split_dataset(chunk)
    sets = normalization(sets)

    # Store the sets in the processed data
    now = datetime.now().timestamp()
    os.mkdir(os.path.join(Config.DATA_DIR, "prepared", str(now)))
    _ = [np.save(os.path.join(Config.DATA_DIR, "prepared", str(now), f"{key}_{now}.npy"), arr) for key, arr in zip(Config.KEYS, sets)]
    print("Store processed dataset chunk /data/processed ...")
    return sets


if __name__ == "__main__":

    _ = process_data()



