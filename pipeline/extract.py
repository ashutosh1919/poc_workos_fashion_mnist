# This script downloads the dataset locally, loads only a chunk of the dataset for the training.

import os
import numpy as np
from tensorflow.keras import datasets
from pipeline.config import Config


def fetch_data():
    """
    Fetches the fashion mnist data from the tf datasets and stores it locally as compressed numpy arrays
    """
    print("Fetching Fashion MNIST Data ...")

    if not os.path.exists(os.path.join(Config.DATA_DIR, "raw_data")):
        os.mkdir(os.path.join(Config.DATA_DIR, "raw_data"))
        # Load fashion mnist
        (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
        # Create directory raw_data and store data locally
        _ = [np.save(os.path.join(Config.DATA_DIR, "raw_data", f"{key}.npy"), item) for key,item in
             zip(Config.KEYS, [x_train, y_train, x_test, y_test])]
    else:
        # Load fashion mnist
        (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
        # Create directory raw_data and store data locally
        _ = [np.save(os.path.join(Config.DATA_DIR, "raw_data", f"{key}.npy"), item) for key, item in
             zip(Config.KEYS, [x_train, y_train, x_test, y_test])]

    print("Storing the Fashion MNIST Data in /data/raw_data ...")


if __name__ == "__main__":
    fetch_data()