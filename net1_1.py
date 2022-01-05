import numpy as np
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Sequential
from arff import load_from_arff_to_dataframe
from tensorflow.keras.utils import to_categorical
from sys import argv

SHOW_PLOTS = len(argv) > 2 and argv[2] == '--plot'

MAX_Z_SCORE = 3
MAX_Z_SCORE_RANGE = [[-MAX_Z_SCORE, MAX_Z_SCORE], [-MAX_Z_SCORE, MAX_Z_SCORE]]

MAX_HISTOGRAM_ORDER_OF_MAGNITUDE = 3


def build_model(size: int, class_num: int, dim_num: int):
    return Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=(size, size, dim_num), padding="same"),
        MaxPooling2D(),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),

        Flatten(),

        Dense(192, activation='relu'),
        Dropout(0.15),

        Dense(128, activation='relu'),
        Dropout(0.15),

        Dense(class_num, activation='softmax')
    ])


def process_series_data(dataframe) -> np.ndarray:
    data = []
    for series in dataframe.to_numpy():
        series = np.array([y.to_numpy() for y in series]).T
        series = series[~np.any(np.isnan(series), axis=1)]
        data.append(series)
    return np.array(data)


def normalize_zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x)


def normalize_histogram(h: np.ndarray) -> np.ndarray:
    h = np.log(h + 1)
    h = np.clip(h, 0, MAX_HISTOGRAM_ORDER_OF_MAGNITUDE)
    h /= MAX_HISTOGRAM_ORDER_OF_MAGNITUDE
    return h


def generate_image(series, size) -> np.ndarray:
    diff = np.diff(series, axis=0, prepend=[series[0]])
    image = []

    if SHOW_PLOTS:
        fig, axs = plt.subplots(nrows=1, ncols=series.shape[1], sharex=True, figsize=(12, 9/series.shape[1]))
        ax_counter = 0

    for x, y in np.array(tuple(zip(diff.T, series.T))):
        h, xe, ye = np.histogram2d(normalize_zscore(x), normalize_zscore(y), bins=size, range=MAX_Z_SCORE_RANGE)
        h = normalize_histogram(h)
        image.append(h)
        if SHOW_PLOTS:
            X, Y = np.meshgrid(xe, ye)
            im = axs[ax_counter].pcolormesh(X, Y, h, cmap="gray")
            ax_counter += 1

    if SHOW_PLOTS:
        fig.subplots_adjust(right=0.8)
        fig.colorbar(im, cax=fig.add_axes([0.85, 0.15, 0.05, 0.7]))
        plt.show()

    image = np.array(image).T
    return image


def generate_images(series: np.ndarray) -> np.ndarray:
    series = np.array([generate_image(x, size) for x in series])
    return series

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
    size = 30
    dataset = argv[1]

    xtrain, ytrain = load_from_arff_to_dataframe(f"./test_data/{dataset}/{dataset}_TRAIN.arff")
    xtrain = process_series_data(xtrain)
    xtrain = generate_images(xtrain)

    dim_num = xtrain.shape[-1]
    ymap, ytrain, class_num = process_ytrain(ytrain)

    model = build_model(size, class_num, dim_num)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=80, batch_size=96)

    xtest, ytest = load_from_arff_to_dataframe(f"./test_data/{dataset}/{dataset}_TEST.arff")
    xtest = process_series_data(xtest)
    xtest = generate_images(xtest)

    ytest = process_ytest(ymap, ytest)

    evaluations = model.evaluate(xtest, ytest)
    print(evaluations)
