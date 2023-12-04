import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
import os
from skimage.io import imshow
import math


# reading dataset
def read_data(path):
    return pd.read_csv(path)


# visualize image keypoints in an image
def visualize_points_img(img, points):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    plt.imshow(img)

    for i in range(0, len(points), 2):
        x_norm = (points[i] + 0.5) * 96
        y_norm = (points[i+1] + 0.5) * 96
        circ = Circle((x_norm, y_norm), 1, color='r')
        ax.add_patch(circ)


# check if a certain keypoints has null value or not
def has_nan_points(keypoints):
    for i in range(len(keypoints)):
        if math.isnan(keypoints.iloc[i]):
            return True

    return False


# collect the data from dataset
def collect_data(data):

    training_images = []
    training_labels = []

    for i in range(len(data)):
        points = data.iloc[i, :-1]
        if has_nan_points(points) is False:
            img = data.iloc[i, -1]        # Get the image data
            img = np.array(img.split(' ')).astype(int)
            # Reshape into an array of size 96x96
            img = np.reshape(img, (96, 96))
            img = img/255         # Normalize image

            keypoints = data.iloc[i, :-1].astype(int).values
            keypoints = keypoints/96 - 0.5  # Normalize keypoint coordinates

            training_images.append(img)
            training_labels.append(keypoints)
        print(f"{i} files done!!")

    return np.array(training_images), np.array(training_labels)


# data augmentation
def augment(img, points):

    flipped_img = img[:, ::-1]

    for i in range(0, len(points), 2):
        x_denormalized = (points[i] + 0.5) * 96
        dx = x_denormalized - 48                # midpoint distance. so half of 96
        x_denormalized_flipped = x_denormalized - 2 * dx
        points[i] = x_denormalized_flipped/96 - 0.5  # normalizing x-coordinate

    return flipped_img, points


def augment_data(images, labels):
    aug_img = []
    aug_label = []

    for i, img in enumerate(images):
        flipped_img, flipped_label = augment(img, labels[i])
        aug_img.append(flipped_img)
        aug_label.append(flipped_label)

    return np.array(aug_img), np.array(aug_label)


def combine_augmented_and_original_data(images, label, aug_img, aug_label):

    final_image = np.concatenate((images, aug_img), axis=0)
    final_label = np.concatenate((label, aug_label), axis=0)

    return final_image, final_label


def get_train_data(data_path):
    data = read_data(data_path)

    data, labels = collect_data(data)

    aug_data, aug_labels = augment_data(data, labels)

    final_data, final_label = combine_augmented_and_original_data(
        data, labels, aug_data, aug_labels)

    return final_data, final_label
