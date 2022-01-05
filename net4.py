import matplotlib.pyplot as plt
import numpy as np
from keras.layers import *
from keras.models import Sequential
from numpy import isnan
from skimage.transform import resize
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


def scale(series, width):
    return resize(series, (width, series.shape[1]))


def generate_image(series, width, *args):
    dimensions = len(series)
    array = np.empty([1, width, dimensions], np.float64)

    series = remove_nans(series)
    series = scale(series, width)

    length = series.shape[0]

    velocities = []

    for d in range(dimensions):
        velocity = []
        for i in range(length - 1):
            v = series[i+1, d] - series[i, d]
            velocity.append(v)
        velocities.append(velocity)

    for d in range(dimensions):
        velocity = velocities[d]

        for i in range(len(velocity)):
                array[0, i, d] = velocity[i]

    if SHOW_PLOTS:
        fig, axs = plt.subplots(nrows=1, ncols=series.shape[1], sharex=True, figsize=(12, 9/series.shape[1]))
        for d in range(dimensions):
            im = axs[d].pcolormesh(array[:, :, d], cmap="gray")
        fig.subplots_adjust(right=0.8)
        fig.colorbar(im, cax=fig.add_axes([0.85, 0.15, 0.05, 0.7]))
        plt.show()

    return array


def build_model(width, class_num, dim_num):
    return Sequential([
        Conv2D(64, (1, 3), activation='relu', input_shape=(1, width, dim_num)),
        MaxPooling2D((1,2)),

        Conv2D(64, (1, 3), activation='relu'),
        MaxPooling2D((1,2)),

        Conv2D(128, (1, 3), activation='relu'),
        MaxPooling2D((1,2)),
        
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
    width = 400
    dataset = argv[1]

    xtrain, ytrain = load_from_arff_to_dataframe(f"./test_data/{dataset}/{dataset}_TRAIN.arff")
    xtrain = [generate_image(x, width) for x in xtrain.values.tolist()]
    xtrain = np.array(xtrain)

    ymap, ytrain, class_num = process_ytrain(ytrain)

    model = build_model(width, class_num, xtrain.shape[3])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=100, batch_size=128)

    xtest, ytest = load_from_arff_to_dataframe(f"./test_data/{dataset}/{dataset}_TEST.arff")
    xtest = [generate_image(x, width) for x in xtest.values.tolist()]
    xtest = np.array(xtest)

    ytest = process_ytest(ymap, ytest)
    
    evaluations = model.evaluate(xtrain, ytrain)
    evaluations = model.evaluate(xtest, ytest)
    print(evaluations)