import os
import csv
import cv2
import json
import numpy as np
import sklearn
from keras.utils.np_utils import to_categorical
from scipy.stats.mstats import mquantiles
from sklearn.model_selection import train_test_split

from keras.layers import Convolution2D, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, Dropout
from keras.models import Model, Sequential


## Collect the data for processing.
samples = []
driving_logs = ['hard-2-back', '1-back', 'gentle-driving']
for k, log in enumerate(driving_logs):
    with open('./data/' + log + '/driving_log.csv') as csvfile:
        temp_samples = []
        reader = csv.reader(csvfile)
        for line in reader:
            temp_samples.append(line)
            if driving_keys[k]:
    samples.extend(temp_samples)

train_samples, validation_samples = train_test_split(samples, shuffle=True, test_size=0.2)
train_sample_size = len(train_samples) * 9
validation_sample_size = len(validation_samples) * 9

def generator(samples, directory_list, batch_size=32,
              img_size_y=160, img_size_x=320, num_channels=3):
    ## Returns a lists of numpy arrays that are of shape:
    ##      (batch_size, img_size_x, img_size_y, depth)
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        ##shuffle(samples)
        for offset in range(0, num_samples - batch_size, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                dir_data = batch_sample[0].split('/')[-3]
                name = './data/' + dir_data + '/IMG/'+batch_sample[0].split('/')[-1]
                left_name = name.replace('center', 'left')
                right_name = name.replace('center', 'right')
                ## Load image.
                center_image = cv2.imread(name)
                ## Convert image to RGB.
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_image_flipped = np.fliplr(center_image)
                ## Get current frames angle.
                center_angle = float(batch_sample[3])
                center_angle_flipped = -center_angle

                ## Generate left, right, flipped left, and flipped right images.
                left_image = cv2.imread(left_name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_image_flipped = np.fliplr(left_image)
                #print(center_image)
                left_angle = center_angle + .6
                left_angle_flipped = -left_angle

                right_image = cv2.imread(right_name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_image_flipped = np.fliplr(right_image)
                #print(center_image)
                right_angle = center_angle -.6
                right_angle_flipped = -right_angle

                images.append(center_image)
                images.append(center_image_flipped)
                images.append(left_image)
                images.append(left_image_flipped)
                images.append(right_image)
                images.append(right_image_flipped)

                angles.append(center_angle)
                angles.append(center_angle_flipped)
                angles.append(left_angle)
                angles.append(left_angle_flipped)
                angles.append(right_angle)
                angles.append(right_angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            ## Return all images and angles.
            yield sklearn.utils.shuffle(X_train, y_train)

batch_data_size = 128

# compile and train the model using the generator function
train_generator = generator(train_samples, driving_logs,
                            batch_size=batch_data_size)
validation_generator = generator(validation_samples, driving_logs,
                                 batch_size=batch_data_size)

## Image shape data.
ch, row, col = 3, 160, 320

classification_model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
classification_model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
classification_model.add(AveragePooling2D())
classification_model.add(Convolution2D(24,5,5,activation="relu", subsample=(2,2)))
classification_model.add(Convolution2D(36, 5, 5, activation="relu", subsample=(2,2)))
classification_model.add(Convolution2D(48, 5, 5, activation="relu", subsample=(2,2)))
classification_model.add(Convolution2D(64, 3, 3, activation="relu"))
classification_model.add(Convolution2D(64, 3, 3, activation="relu"))
classification_model.add(Flatten())
classification_model.add(Dense(1024, activation="relu"))
classification_model.add(Dropout(0.5))
classification_model.add(Dense(100))
classification_model.add(Dense(50))
classification_model.add(Dense(10))
classification_model.add(Dense(1))
classification_model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

classification_model.fit_generator(train_generator, samples_per_epoch=train_sample_size,
            validation_data=validation_generator,
            nb_val_samples=validation_sample_size, nb_epoch=5, max_q_size=1)

classification_model.save("model.h5")
