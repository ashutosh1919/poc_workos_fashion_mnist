import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import yaml
import pickle
import os
import json
import sys

params = yaml.safe_load(open('params.yaml'))['evaluate']
train_params = yaml.safe_load(open('params.yaml'))['train']

if len(sys.argv)!=4:
    print("Not passed enough arguements to evaluate stage.")
    sys.exit(1)

print("Starting evaluation stage...")

verbose = params['seed']
class_names = params['class_names']

n_classes = train_params['n_classes']
epochs = train_params['epochs']
optimizer = train_params['optimizer']
metrics = train_params['metrics']

data_path = sys.argv[1]
model_code_dir = sys.argv[2]
model_ckpt_dir = sys.argv[3]
os.makedirs(os.path.join('.', 'results'), exist_ok=True)
out_file = os.path.join("results", "metrics.json")
out_confusion_png = os.path.join("results", "confusion_matrix.png")


def load_data(pkl_filepath):
    with open(pkl_filepath, "rb") as f:
        data = pickle.load(f)
    return data["images"], data["labels"]


def get_true_labels(labels):
    y = []
    for i in range(labels.shape[0]):
        index = np.argmax(labels[i])
        y.append(class_names[index])
    return np.array(y)


def plot_cm(y_true, y_pred, filename, figsize=(10,10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plot = sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    figure = plot.get_figure()
    figure.savefig(filename)

sys.path.append(model_code_dir)
from model import *

images, labels = load_data(data_path)
restored_model = LeNet(images[0].shape, n_classes, optimizer, metrics)
restored_model.load_weights(model_ckpt_dir)

preds = restored_model.predict(images)
y_pred = get_true_labels(preds)
y_true = get_true_labels(labels)
plot_cm(y_true, y_pred, out_confusion_png)

loss, *eval_list = restored_model.evaluate(images, labels, verbose=2)

print("Evaluation results: ")
eval_dict = {}
for mt_key, mt_val in zip(metrics, eval_list):
    eval_dict.update({mt_key: mt_val})
    print(f"{mt_key}: ", mt_val)

with open(out_file, "w") as f:
    json.dump(eval_dict, f)

print("Evaluation stage completed...")