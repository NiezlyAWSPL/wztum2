import numpy as np
from keras.models import Sequential
from sktime.utils.data_io import load_from_arff_to_dataframe
import matplotlib.pyplot as plt
from keras.layers import *
from numpy import NaN, floor, isnan, mean, log, uint8
from tensorflow.keras.utils import to_categorical
from scipy import interpolate
from skimage.transform import resize



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
    array = np.empty([1, width, dimensions], np.int32)

    series = remove_nans(series)
    series = scale(series, width)
    
    for j in range(width):
        for dim in range(dimensions):
            array[0,j,dim] = series[j, dim]

    # plt.imshow(array.astype(uint8), interpolation='nearest')
    # plt.show()

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


if __name__ == "__main__":
    width = 200

    xtrain, ytrain = load_from_arff_to_dataframe('./test_data/CharacterTrajectories/CharacterTrajectories_TRAIN.arff')
    xtrain = [generate_image(x, width) for x in xtrain.values.tolist()]
    xtrain = np.array(xtrain)

    ytrain = np.array([int(x) - 1 for x in ytrain])
    class_num = max(ytrain) + 1
    ytrain = to_categorical(ytrain, class_num)

    model = build_model(width, class_num, xtrain.shape[3])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=100, batch_size=128)

    xtest, ytest = load_from_arff_to_dataframe('./test_data/CharacterTrajectories/CharacterTrajectories_TEST.arff')
    xtest = [generate_image(x, width) for x in xtest.values.tolist()]
    xtest = np.array(xtest)

    ytest = np.array([int(x) - 1 for x in ytest])
    ytest = to_categorical(ytest, class_num)
    
    evaluations = model.evaluate(xtrain, ytrain)
    evaluations = model.evaluate(xtest, ytest)
    print(evaluations)