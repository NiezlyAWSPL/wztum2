import numpy as np
from keras.models import Sequential
from sktime.utils.data_io import load_from_arff_to_dataframe
import matplotlib.pyplot as plt
from keras.layers import *
from numpy import floor, isnan, mean, log
from tensorflow.keras.utils import to_categorical


def generate_image(series, width, *args):
    array = np.empty([width, width, 1], np.int32)
    series = list(filter(lambda s: not isnan(s), series))
    lenght = len(series)
    widthScale = lenght / width

    series = [mean(series[int(i * widthScale):int((i + 1) * widthScale)]) for i in range(width)]
    
    for i in range(width):
        for j in range(width):
            x = series[i] * series[j]
            array[i,j,0] = x

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
        Dense(256, activation='relu'),
        Dropout(0.1),
        Dense(class_num, activation='softmax')
    ])


if __name__ == "__main__":
    width = 35
    height = 35

    xtrain, ytrain = load_from_arff_to_dataframe('./test_data/CharacterTrajectories/CharacterTrajectoriesDimension1_TRAIN.arff')
    xtrain = [generate_image(x.values.tolist(), width) for x in xtrain['dim_0']]
    xtrain = np.array(xtrain)

    ytrain = np.array([int(x) - 1 for x in ytrain])
    class_num = max(ytrain) + 1
    ytrain = to_categorical(ytrain, class_num)

    model = build_model(width, height, class_num)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=150, batch_size=128)


    xtest, ytest = load_from_arff_to_dataframe('./test_data/CharacterTrajectories/CharacterTrajectoriesDimension1_TEST.arff')
    xtest = [generate_image(x.values.tolist(), width) for x in xtest['dim_0']]
    xtest = np.array(xtest)

    ytest = np.array([int(x) - 1 for x in ytest])
    ytest = to_categorical(ytest, class_num)
    
    evaluations = model.evaluate(xtrain, ytrain)
    evaluations = model.evaluate(xtest, ytest)
    print(evaluations)