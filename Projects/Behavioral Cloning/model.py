import os
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        try:
            float(line[3])
        except ValueError:
            continue
        lines.append(line)
        copy = list(line)
        copy.append("flip")
        lines.append(copy)
with open('newdata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        try:
            float(line[3])
        except ValueError:
            continue
        lines.append(line)
        copy = list(line)
        copy.append("flip")
        lines.append(copy)
with open('recoverydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        try:
            float(line[3])
        except ValueError:
            continue
        lines.append(line)
        copy = list(line)
        copy.append("flip")
        lines.append(copy)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                last_ele = batch_sample[-1]
                flipped = last_ele == "flip"
                old_path = batch_sample[0]
                path_comps = old_path.split('/')
                if len(path_comps) <= 2:
                    folder = 'data'
                else:
                    folder = path_comps[-3]
                filename = path_comps[-1]
                new_path = folder+'/IMG/'+filename
                center_image = cv2.imread(new_path)
                center_angle = float(batch_sample[3])
                if flipped:
                    center_image = cv2.flip(center_image, 1)
                    center_angle = center_angle * -1.
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch)),
        output_shape=(row, col, ch)))
model.add(Convolution2D(24, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')