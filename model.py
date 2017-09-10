import os
import csv
import cv2
import numpy as np
import sklearn

samples_list = []
driving_logs = ['1-back', 'gentle-driving', 'hard-1', 'hard-2-back']
## The keys are to say if you should reverse it.
driving_keys = [True, True, False, False]
steering_angles = []
for k, log in enumerate(driving_logs):
    with open('./data/' + log + '/driving_log.csv') as csvfile:
        temp_samples = []
        reader = csv.reader(csvfile)
        for line in reader:
            temp_samples.append(line)
            steering_angles.append(float(line[3]))
    samples_list.append(temp_samples)

## Calculate the histogram.
angle_ranges = np.histogram(np.array(steering_angles), bins='auto', range=(-1.0, 1.0))
## TODO: Make a dictionary mapping histogram to value and value to average histogram.
## Take the center value of the histogram's two values to produce the value to average histogram.
## -3, -2, -1, 0, 1 2 3
## -3, -2, -1, 1, 2, 3
half_hist = len(angle_ranges[0]) // 2
hist_to_index = angle_ranges[1]
index_to_angle = {}
index_to_angle[0] = -1.0
index_to_angle[len(hist_to_index) - 1] = 1.0
index_to_angle[half_hist + 1] = 0.0
index_to_angle[half_hist + 2] = 0.0
for k in range(1, half_hist + 1):
    index_to_angle[k] = (hist_to_index[k] + hist_to_index[k+1]) / 2
for k in range(half_hist + 3, len(hist_to_index) - 1):
    index_to_angle[k] = (hist_to_index[k] + hist_to_index[k + 1]) / 2

#print(index_to_angle[half_hist])
#print(hist_to_index)

## Use the histogram as an output for the neural network.
##print(hist_to_index.size)
output_shape = hist_to_index.size

from sklearn.model_selection import train_test_split
train_sample_list = []
validation_sample_list = []
train_sample_size = 0
validation_sample_size = 0
for samples in samples_list:
    train_samples, validation_samples = train_test_split(samples, test_size=0.2,
                                                     shuffle=False)
    train_sample_list.append(train_samples)
    train_sample_size += len(train_samples)
    validation_sample_list.append(validation_samples)
    validation_sample_size += len(validation_samples)

def generator(samples_list, directory_list, batch_size=32, num_frames=6,
              img_size_y=160, img_size_x=320, num_channels=3):
    ## Returns a lists of numpy arrays that are of shape:
    ##      (img_size_x, img_size_y, depth * num_frames)
    #num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        ##shuffle(samples)
        for key, samples in enumerate(samples_list):
            num_samples = len(samples)
            for offset in range(0, num_samples - batch_size - num_frames, batch_size + num_frames):
                batch_samples = samples[offset:offset+batch_size + num_frames]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    name = './data/' + directory_list[key] + '/IMG/'+batch_sample[0].split('/')[-1]
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
                    temp_image_np = np.zeros((img_size_y, img_size_x, num_channels * num_frames))
                    ## Get slices.
                    temp_image = images[i:i+num_frames]
                    ## Reshape slices to be of the proper shape.
                    temp_image = np.stack(temp_image)
                    for i in range(num_frames):
                        temp_image_np[:, :, i * num_channels: i * num_channels + num_channels] = temp_image[i]
                    image_sets.append(temp_image_np)
                    #print(temp_image.shape)
                    #temp_image = temp_image.reshape(
                    #    (img_size_y, img_size_x, num_channels * num_frames))
                    #print(temp_image.shape)
                    #image_sets.append(temp_image)
                X_train = np.stack(image_sets)
                #print(X_train.shape)
                ## Only use the last of each set of frames for predicting.
                y_train = np.array(angles[num_frames:])

                # trim image to only see section with road
                #X_train = np.array(images)
                #y_train = np.array(angles)
                #print(X_train)
                yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_sample_list, driving_logs,
                            batch_size=4, num_frames=6)
validation_generator = generator(validation_sample_list, driving_logs,
                                 batch_size=4, num_frames=6)

#print(next(train_generator)[0].shape)

num_frames = 6
ch, row, col = 3 * num_frames, 160, 320  # Trimmed image format *not really*
#ch, row, col = 3, 160, 320

from keras.layers import Conv2D, MaxPooling2D, Input, Lambda, merge, Dense, Flatten
from keras.models import Model, Sequential

input_data = Input(shape=(row, col, ch), name="Main_Input")
'''
reg_lambda = Lambda(lambda x: x/127.5 - 1.,
                   input_shape=(row, col, ch),
                   output_shape=(row, col, ch))
input_img = reg_lambda(input_data)
'''

tower_1 = Conv2D(32, 1, 1, border_mode='same', activation='relu')(input_data)
tower_1 = Conv2D(32, 3, 3, border_mode='same', activation='relu')(tower_1)

tower_2 = Conv2D(32, 1, 1, border_mode='same', activation='relu')(input_data)
tower_2 = Conv2D(32, 5, 5, border_mode='same', activation='relu')(tower_2)

#tower_3 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input_img)
#tower_3 = Conv2D(64, 1, 1, border_mode='same', activation='relu')(tower_3)

concat_layer = merge([tower_1, tower_2], mode='concat', concat_axis=1,
                                  name="Main_Output")



flat_layer = Flatten()(concat_layer)

#fc1 = Dense(256, activation='relu')(flat_layer)

fc1 = Dense(128, activation='relu')(flat_layer)

fc1 = Dense(16, activation='relu')(fc1)

output = Dense(1)(flat_layer)

regression_model = Model(input=input_data, output=output)

#model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
#model.add(Lambda(lambda x: x/127.5 - 1.,
#        input_shape=(ch, row, col),
#        output_shape=(ch, row, col)))
#model.add(... finish defining the rest of your model architecture here ...)

#regression_model = Sequential()
#regression_model.add(Flatten(input_shape=(160,320,18)))
#regression_model.add(Dense(1))

regression_model.compile(loss='mse', optimizer='adam')

#batch_data = next(train_generator)
#regression_model.train_on_batch(batch_data[0], batch_data[1])


#regression_model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
#            validation_data=validation_generator,
#            nb_val_samples=len(validation_samples), nb_epoch=3, max_q_size=1)
regression_model.fit_generator(train_generator, samples_per_epoch=train_sample_size,
            validation_data=validation_generator,
            nb_val_samples=validation_sample_size, nb_epoch=3, max_q_size=1)

regression_model.save("model.h5")
