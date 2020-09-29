import os


class Config:

    KEYS = ["x_train", "y_train", "x_test", "y_test"]
    DATA_DIR = os.path.join(os.path.abspath("../"), "data")
    MODEL_DIR = os.path.join(os.path.abspath("../"), "model")
    TRAIN_SIZE = 10000
    TEST_SIZE = 500
    NUM_CLASSES = 10
    EPOCHS = 10
