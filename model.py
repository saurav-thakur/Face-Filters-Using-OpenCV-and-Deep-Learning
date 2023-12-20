import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from read_data import get_train_data, visualize_points_img

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint


# print(label)


def my_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2,
              padding='same', input_shape=(96, 96, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(64, kernel_size=3, strides=2,
              padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(128, kernel_size=3, strides=2,
              padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(64, kernel_size=3, strides=2,
              padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(16, kernel_size=3, strides=2,
              padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(30, activation="linear"))

    model.compile(optimizer='adam',
                  loss="mean_absolute_error", metrics=["mse"])

    return model


def model_training(model):
    checkpoint = ModelCheckpoint(
        filepath="weights/checkpoint-{epoch:02d}.hdf5")
    epochs = 10
    history = model.fit(data, label, epochs=epochs, validation_split=0.2,
                        batch_size=100, callbacks=[checkpoint])

    history_dict = history.history

    loss = history_dict['loss']
    mse = history_dict['mse']
    val_mse = history_dict['val_mse']
    val_loss = history_dict['val_loss']

    # # Access accuracy and loss values from history
    epoch_numbers = range(1, epochs + 1)
    csv_columns = ['epoch', 'loss',
                   'mse', 'val_mse', 'val_loss']
    csv_data = zip(epoch_numbers, loss, mse, val_mse, val_loss)

    csv_file_path = 'model_tracking/training_metrics.csv'
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_columns)
        writer.writerows(csv_data)

    print(f"Training metrics saved to {csv_file_path}")


def loading_trained_model(model):
    model.load_weights('weights/checkpoint-200.hdf5')


def model_testing(model, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1, 96, 96, 1))
    prediction = model.predict(img)
    visualize_points_img(img=img, points=prediction[0])


if __name__ == "__main__":
    path = "../facial-keypoints-detection/training/training.csv"
    data, label = get_train_data(path)
    model = my_model()
    model_training(model)
