import numpy as np
from keras.layers import *
from keras.models import Sequential
from matplotlib import pyplot as plt
from numpy import isnan
from arff import load_from_arff_to_dataframe
from tensorflow.keras.utils import to_categorical
from sys import argv


SHOW_PLOTS = len(argv) > 2 and argv[2] == '--plot'


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

    if SHOW_PLOTS:
        fig, axs = plt.subplots(nrows=1, ncols=series.shape[1], sharex=True, figsize=(12, 9/series.shape[1]))
        for d in range(dimensions):
            im = axs[d].pcolormesh(array[:, :, d], cmap="gray")
        fig.subplots_adjust(right=0.8)
        fig.colorbar(im, cax=fig.add_axes([0.85, 0.15, 0.05, 0.7]))
        plt.show()

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
    width = 34
    height = 34
    color_scale = 40

    dataset = argv[1]

    xtrain, ytrain = load_from_arff_to_dataframe(f"./test_data/{dataset}/{dataset}_TRAIN.arff")
    width = width //2 * 2 + 1
    height = height //2 * 2 + 1
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