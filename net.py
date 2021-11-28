import numpy as np
import pandas as pd
from keras.models import Sequential
from sktime.utils.data_io import load_from_arff_to_dataframe
import matplotlib.pyplot as plt
from keras.layers import *
from numpy import floor, isnan, mean, log, uint8
from tensorflow.keras.utils import to_categorical

def remove_nans(series):
    result = []
    for i in range(len(series[0])):
        no_nans = True
        l = []
        for j in range(len(series)):
            v = series[j][i]
            if isnan(v):
                no_nans = False
                break
            l.append(v)
        if no_nans:
            result.append(l)
    return np.array(result)

def generate_image(series, width, height, color_scale):
    dimensions = len(series)
    array = np.zeros([height, width, dimensions], np.int32)

    series = remove_nans(series)
    length = series.shape[0]

    max_velocity = 0
    max_acceleration = 0

    velocities = []
    accelerations = []

    for d in range(dimensions):
        velocity = []
        acceleration = []
        for i in range(length - 1):
            v = series[i+1, d] - series[i, d]
            if abs(v) > max_velocity:
                max_velocity = abs(v)
            velocity.append(v)
        for i in range(len(velocity) - 1):
            a = velocity[i+1] - velocity[i]
            if abs(a) > max_acceleration:
                max_acceleration = abs(a)
            acceleration.append(a)
        velocities.append(velocity)
        accelerations.append(acceleration)

    half_width = int(width / 2)
    half_height = int(height / 2)
    v_scale = half_width / max_velocity
    a_scale = half_height / max_acceleration

    for d in range(dimensions):
        velocity = velocities[d]
        acceleration = accelerations[d]
        for i in range(len(acceleration)):
            velocity[i] = int(velocity[i] * v_scale + half_width)
            acceleration[i] = int(acceleration[i] * a_scale + half_height)

        for i in range(len(acceleration)):
            if array[velocity[i], acceleration[i], d] < color_scale:
                array[velocity[i], acceleration[i], d] += 1

    return array

def build_model(width, height, class_num, dim_num):
    return Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(width, height, dim_num)),
        MaxPooling2D(),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.1),
        Dense(class_num, activation='softmax')
    ])

def process_ytrain(ytrain):
    ymap = {}
    yarray = []
    num = 0
    for y in ytrain:
        if y not in ymap:
            ymap[y] = num
            num += 1
        yarray.append(ymap[y])

    return (ymap, to_categorical(np.array(yarray)), num)

def process_ytest(ymap, ytest):
    return to_categorical(np.array([ymap[v] for v in ytest]))

if __name__ == "__main__":
    width = 33
    height = 33
    color_scale = 40

    dataset = "RefrigerationDevices"

    xtrain, ytrain = load_from_arff_to_dataframe(f"./test_data/{dataset}/{dataset}_TRAIN.arff")
    xtrain = [generate_image(x, width, height, color_scale) for x in xtrain.values.tolist()]
    xtrain = np.array(xtrain)

    ymap, ytrain, class_num = process_ytrain(ytrain)

    model = build_model(width, height, class_num, xtrain.shape[3])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=100, batch_size=128)

    xtest, ytest = load_from_arff_to_dataframe(f"./test_data/{dataset}/{dataset}_TEST.arff")
    xtest = [generate_image(x, width, height, color_scale) for x in xtest.values.tolist()]
    xtest = np.array(xtest)

    ytest = process_ytest(ymap, ytest)
    
    evaluations = model.evaluate(xtrain, ytrain)
    evaluations = model.evaluate(xtest, ytest)
    print(evaluations)