import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import yaml
import pickle
import os
import sys

params = yaml.safe_load(open('params.yaml'))['evaluate']

if len(sys.argv)!=4:
    print("Not passed enough arguements to evaluate stage.")
    sys.exit(1)

print("Starting evaluation stage...")

verbose = params['seed']

data_path = sys.argv[1]
model_code_dir = sys.argv[2]
model_ckpt_dir = sys.argv[3]

def load_data(pkl_filepath):
    with open(pkl_filepath, "rb") as f:
        data = pickle.load(f)
    return data["images"], data["labels"]

sys.path.append(model_code_dir)
from model import *
restored_model = LeNet()

images, labels = load_data(data_path)
restored_model.load_weights(model_ckpt_dir)
loss, acc, mse = restored_model.evaluate(images, labels, verbose=2)
print("Evaluation results: ")
print("Loss: ", loss)
print("Accuracy: ", acc)
print("MSE", mse)
print("Evaluation stage completed...")