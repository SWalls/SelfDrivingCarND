import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

USE_FLIPPED = True # If True, will add flipped images and angles to training data set

samples = [] # Array of training data

# Load the training data samples from the csv and files.
def load_samples(data_dir):
    global samples
    with open(data_dir+"/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            try:
                float(line[3])
            except ValueError:
                continue
            samples.append(line)
            # Add the flipped image to the data set, and add the 
            # "flip" flag so that we know to flip the angle later.
            if USE_FLIPPED:
                copy = list(line)
                copy.append("flip")
                samples.append(copy)

load_samples('data') # center lane driving in track one (provided)
load_samples('newdata') # more center lane driving in track one
# load_samples('reversedata') # driving in reverse around track one
# load_samples('recoverydata3') # recovering from left and right side of track one
# load_samples('recoverydata4') # recovering from left and right side of track one
# load_samples('recoverydata5') # recovering from left and right side of track one
# load_samples('recoverydata6') # recovering from left and right side of track one
# load_samples('track2data') # center lane driving in track two

# split the dataset into test and validation data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                flipped = batch_sample[-1] == "flip" # check to see if the "flip" flag is present
                # The file path in the csv file is incorrect.
                # But we can use this path to make the correct path to the image file.
                old_path = batch_sample[0]
                path_comps = old_path.split('/')
                if len(path_comps) <= 2:
                    folder = 'data'
                else:
                    folder = path_comps[-3]
                filename = path_comps[-1]
                new_path = folder+'/IMG/'+filename
                # Finally read in the image using the correct path.
                center_image = cv2.imread(new_path)
                center_angle = float(batch_sample[3])
                if flipped: # Flip the image and angle if it should be flipped.
                    center_image = cv2.flip(center_image, 1)
                    center_angle *= -1.
                # Add the image and angle to the set of data and labels, respectively.
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

# Create the model (based on NVIDIA architecture, but with only 4 conv layers)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)))) # trim image to only see section with road
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
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=2)

model.save('model-b.h5')
