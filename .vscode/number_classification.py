import matplotlib.pyplot as plt
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import SGD
from keras.datasets import mnist

from sklearn import datasets

from image_processing import scale_to_range, resize_image

def prepare_images_for_cnn(X_images):
    
    numbers = []

    for img in X_images:
        #isecanje okvira
        #threshold, threshold_image = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)

        mask = img > 0
        coords = np.argwhere(mask)

        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1

        new_image = img[x0:x1, y0:y1]

        #promena velicine u 28x28 + skaliranje u [0,1]
        new_image = scale_to_range(resize_image(new_image))

        numbers.append(new_image)
        
    numbers = np.array(numbers, np.float32)

    #za CNN
    numbers = numbers.reshape(numbers.shape[0],28,28,1)

    return numbers

def result_vector(num):
    num_array = np.zeros(10)
    num_array[num] = 1

    return num_array

def prepare_values_for_cnn(Y_values):
    values = []

    for y in Y_values:
        values.append(result_vector(y))

    values = np.array(values, np.float32)

    return values

def train_cnn():

    #ucitavanje MNIST dataset-a
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #priprema dataset-a za CNN
    X_train = prepare_images_for_cnn(X_train)
    X_test = prepare_images_for_cnn(X_test)

    y_train = prepare_values_for_cnn(y_train)
    y_test = prepare_values_for_cnn(y_test)

    #definisanje arhitekture CNN
    cnn = Sequential()

    cnn.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
    cnn.add(Conv2D(32, kernel_size=3, activation='relu'))
    cnn.add(Flatten())
    cnn.add(Dense(10, activation='softmax'))

    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #treniranje CNN
    cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

    #cuvanje istrenirane CNN u fajl
    cnn.save('cnn.h5')

    return cnn

def classification(number_image, cnn):   
    result = cnn.predict(number_image)

    #vraca indeks maksimalnog broja u nizu
    return max(enumerate(result[0]), key=lambda x: x[1])[0]