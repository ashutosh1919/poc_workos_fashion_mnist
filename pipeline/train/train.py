from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import json
import yaml
import pickle
import os
import sys


params = yaml.safe_load(open('params.yaml'))['train']

if len(sys.argv)!=5:
    print("Not passed enough arguements to train stage.")
    sys.exit(1)

print("Starting the training stage...")

n_classes = params['n_classes']
epochs = params['epochs']
optimizer = params['optimizer']
metrics = params['metrics']

os.makedirs(os.path.join("model", "LeNet_checkpoints"), exist_ok=True)
output_model = os.path.join("model", "LeNet_checkpoints", "best.ckpt")

os.makedirs(os.path.join('.', 'results'), exist_ok=True)
out_file_loss = os.path.join("results", "train_stats_loss.json")
out_file_acc = os.path.join("results", "train_stats_acc.json")
out_file_mse = os.path.join("results", "train_stats_mse.json")

input_train = sys.argv[1]
input_test = sys.argv[2]
input_result_schema = sys.argv[3]
model_code_dir = sys.argv[4]

cb_res_loss = pd.DataFrame(columns=["train_loss", "val_loss"])
cb_res_acc = pd.DataFrame(columns=["train_acc", "val_acc"])
cb_res_mse = pd.DataFrame(columns=["train_mse", "val_mse"])

def load_data(pkl_filepath):
    with open(pkl_filepath, "rb") as f:
        data = pickle.load(f)
    return data["images"], data["labels"]

class StatReaderCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            cb_res_loss.loc[epoch] = [logs["loss"], logs["val_loss"]]
            cb_res_acc.loc[epoch] = [logs["accuracy"], logs["val_accuracy"]]
            cb_res_mse.loc[epoch] = [logs["mse"], logs["val_mse"]]
        except:
            print("Error Extracting Values at the end of epoch.")

def restructure_df(res, out_file):
    cols = res.columns
    with open(input_result_schema, "r") as sf:
        schema = json.loads(sf.read())
    schema["title"] = cols[0].split('_')[-1].capitalize() + " Comparision"
    for i in range(res.shape[0]):
        for j in range(2):
            obj = {
                "Parameter": cols[j],
                "Epoch": i,
                "Value": res.loc[i, cols[j]]
            }
            schema["data"]["values"].append(obj)
    schema["encoding"]["x"] = { "field": "Epoch", "type": "quantitative", "title": "Epoch" }
    schema["encoding"]["y"] = { "field": "Value", "type": "quantitative", "title": "Value", "scale": {"zero": False} }
    schema["encoding"]["color"] = { "field": "Parameter", "type": "nominal" }
    text = json.dumps(schema)
    with open(out_file, "w") as f:
        f.write(text)

x_train, y_train = load_data(input_train)
x_test, y_test = load_data(input_test)

sys.path.append(model_code_dir)
from model import LeNet

model = LeNet(x_train[0].shape, n_classes, optimizer, metrics)
checkpoint = ModelCheckpoint(output_model, 
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='max')
callbacks_list = [checkpoint, StatReaderCallback()]
model.fit(x=x_train, 
          y=y_train, 
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose=1,
          callbacks=callbacks_list)

restructure_df(cb_res_loss, out_file_loss)
restructure_df(cb_res_acc, out_file_acc)
restructure_df(cb_res_mse, out_file_mse)

print("Training stage completed...")