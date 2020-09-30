from tensorflow.keras.callbacks import ModelCheckpoint
import yaml
import pickle
import os
import sys


params = yaml.safe_load(open('params.yaml'))['train']

if len(sys.argv)!=4:
    print("Not passed enough arguements to train stage.")
    sys.exit(1)

print("Starting the training stage...")

n_classes = params['n_classes']
epochs = params['epochs']
optimizer = params['optimizer']
metrics = params['metrics']

os.makedirs(os.path.join("model", "LeNet_checkpoints"), exist_ok=True)
output_model = os.path.join("model", "LeNet_checkpoints", "best.ckpt")

input_train = sys.argv[1]
input_test = sys.argv[2]
model_code_dir = sys.argv[3]


def load_data(pkl_filepath):
    with open(pkl_filepath, "rb") as f:
        data = pickle.load(f)
    return data["images"], data["labels"]


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
callbacks_list = [checkpoint]
model.fit(x=x_train, 
          y=y_train, 
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose=1,
          callbacks=callbacks_list)

print("Training stage completed...")