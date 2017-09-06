import os
import csv

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2,
                                                     shuffle=False)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32, num_frames=6,
              img_size_y=160, img_size_x=320, num_channels=3):
    ## Returns a lists of numpy arrays that are of shape:
    ##      (img_size_x, img_size_y, depth * num_frames)
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        ##shuffle(samples)
        for offset in range(0, num_samples, batch_size + num_frames):
            batch_samples = samples[offset:offset+batch_size + num_frames]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                #print(name)
                center_image = cv2.imread(name)
                #print(center_image)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            ## TODO: Make batch_size arrays of num_frames in size.
            ##      Reshape them to be:
            ##          (img_size_x, img_size_y, depth * num_frames)
            image_sets = []
            for i in range(len(images) - num_frames):
                ## Get slices.
                temp_image = images[i:i+num_frames]
                ## Reshape slices to be of the proper shape.
                temp_image = np.array(temp_image)
                #print(temp_image.shape)
                temp_image = temp_image.reshape(
                    (img_size_y, img_size_x, num_channels * num_frames))
                #print(temp_image.shape)
                image_sets.append(temp_image)
            X_train = np.stack(image_sets)
            print(X_train.shape)
            ## Only use the last of each set of frames for predicting.
            y_train = np.array(angles[:-num_frames])

            # trim image to only see section with road
            #X_train = np.array(images)
            #y_train = np.array(angles)
            #print(X_train)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples,
                            batch_size=32, num_frames=6)
validation_generator = generator(validation_samples,
                                 batch_size=32, num_frames=6)

#print(next(train_generator)[0].shape)

num_frames = 6
ch, row, col = 3 * num_frames, 160, 320  # Trimmed image format *not really*

from keras.layers import Conv2D, MaxPooling2D, Input, Lambda, merge, Dense, Flatten
from keras.models import Model

input_data = Input(shape=(row, col, ch), name="Main_Input")
reg_lambda = Lambda(lambda x: x/127.5 - 1.,
                   input_shape=(row, col, ch),
                   output_shape=(row, col, ch))
input_img = reg_lambda(input_data)

tower_1 = Conv2D(64, 1, 1, border_mode='same', activation='relu')(input_img)
tower_1 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, 1, 1, border_mode='same', activation='relu')(input_img)
tower_2 = Conv2D(64, 5, 5, border_mode='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input_img)
tower_3 = Conv2D(64, 1, 1, border_mode='same', activation='relu')(tower_3)

concat_layer = merge([tower_1, tower_2, tower_3], mode='concat', concat_axis=1,
                                  name="Main_Output")

flat_layer = Flatten()(concat_layer)

fc1 = Dense(64, activation='relu')(flat_layer)

output = Dense(1)(fc1)

regression_model = Model(input=input_data, output=output)

#model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
#model.add(Lambda(lambda x: x/127.5 - 1.,
#        input_shape=(ch, row, col),
#        output_shape=(ch, row, col)))
#model.add(... finish defining the rest of your model architecture here ...)

regression_model.compile(loss='mse', optimizer='adam')

regression_model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
            validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=3)
