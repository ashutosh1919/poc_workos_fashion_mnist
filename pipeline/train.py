import os
from pipeline.config import Config
from pipeline.prepare import process_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime


class LeNet(Sequential):
    """
    LeNet-5 model class, 2 convolution layers, 2 average pooling, 1 dense layer, activation = tanh
    """
    def __init__(self, input_shape, nb_classes):
        super().__init__()

        self.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape,
                        padding="same"))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Flatten())
        self.add(Dense(120, activation='tanh'))
        self.add(Dense(84, activation='tanh'))
        self.add(Dense(nb_classes, activation='softmax'))

        self.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])


def learn():

    x_train, y_train, x_test, y_test = process_data()
    model = LeNet(x_train[0].shape, Config.NUM_CLASSES)
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    checkpoint = ModelCheckpoint(os.path.join(Config.MODEL_DIR, str(timestamp)), monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit(x_train, y=y_train, epochs=Config.EPOCHS, validation_data=(x_test, y_test), verbose=1, callbacks=callbacks_list)


if __name__ == "__main__":
    learn()
