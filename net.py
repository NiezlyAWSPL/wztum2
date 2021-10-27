import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from keras.models import Sequential
from sktime.utils.data_io import load_from_arff_to_dataframe
import matplotlib.pyplot as plt
from keras.layers import *
from keras.backend import set_session
from numpy import isnan
from tensorflow.keras.utils import to_categorical


def generate_image(series, width, height, color_scale):
    max_velocity = 0
    max_acceleration = 0
    velocity = []
    acceleration = []
    for i in range(len(series) - 1):
        v = series[i+1] - series[i]
        if abs(v) > max_velocity:
            max_velocity = abs(v)
        velocity.append(v)
    for i in range(len(velocity) - 1):
        a = velocity[i+1] - velocity[i]
        if abs(a) > max_acceleration:
            max_acceleration = abs(a)
        acceleration.append(a)

    half_width = int(width / 2)
    half_height = int(height / 2)
    v_scale = half_width / max_velocity
    a_scale = half_height / max_acceleration
    for i in range(len(acceleration)):
        if isnan(velocity[i]) or isnan(acceleration[i]):
            continue
        velocity[i] = int(velocity[i] * v_scale + half_width)
        acceleration[i] = int(acceleration[i] * a_scale + half_height)

    array = np.zeros([height, width, 1], np.int32)
    for i in range(len(acceleration)):
        if isnan(velocity[i]) or isnan(acceleration[i]):
            continue
        if array[velocity[i], acceleration[i], 0] < color_scale:
            array[velocity[i], acceleration[i], 0] += 1

    # plt.imshow(array, cmap="gray_r")
    # plt.show()

    return array


def build_model(width, height, class_num):
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)),
        MaxPooling2D(),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dense(class_num, activation='softmax')
    ])


if __name__ == "__main__":
    width = 33
    height = 33
    color_scale = 40

    xtrain, ytrain = load_from_arff_to_dataframe('./test_data/CharacterTrajectories/CharacterTrajectoriesDimension1_TRAIN.arff')
    xtrain = [generate_image(x.values.tolist(), width, height, color_scale) for x in xtrain['dim_0']]
    xtrain = np.array(xtrain)

    ytrain = np.array([int(x) - 1 for x in ytrain])
    class_num = max(ytrain) + 1
    print(class_num)
    ytrain = to_categorical(ytrain, class_num)

    model = build_model(width, height, class_num)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=200, batch_size=128)


    xtest, ytest = load_from_arff_to_dataframe('./test_data/CharacterTrajectories/CharacterTrajectoriesDimension1_TEST.arff')
    xtest = [generate_image(x.values.tolist(), width, height, color_scale) for x in xtest['dim_0']]
    xtest = np.array(xtest)

    ytest = np.array([int(x) - 1 for x in ytest])
    ytest = to_categorical(ytest, class_num)
    
    evaluations = model.evaluate(xtrain, ytrain)
    evaluations = model.evaluate(xtest, ytest)
    print(evaluations)