#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center_lane_example.jpg "Center Lane Driving"
[image3]: ./examples/left_training.jpg "Recovery Image"
[image4]: ./examples/center_training.jpg "Recovery Image"
[image5]: ./examples/right_training.jpg "Recovery Image"
[image6]: ./examples/training_small.jpg "Normal Image"
[image7]: ./examples/training_small_flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

My model consists of an altered form of the Nvidia architecture (model.py lines 116-134) 

The model includes RELU layers to introduce nonlinearity (code lines 122-128), and the data is normalized in the model using a Keras lambda layer (code lines 118-120). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 129). It was found that placing the dropout layer much later or earlier in the network caused issues. Placing it earlier would block the model from learning a good feature set in the convolutional portion and later would cause the system to find a median value rather than actually learning from the data.

The model was trained and validated on different data sets to ensure that the model was not overfitting. This included using data from the secondary track and using a 20-80 split of validation vs. training data. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 134).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and data from the second track. Due to the general variation within my own driving, the center lane driving included a number of correction maneuvers. Also, the left and right cameras were used to generate extra side driving data.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to have a base layer of feature extractors through convolutional layers followed by a set of fully connected layers for the final predictions.

My first step was to use a convolution neural network model similar to the NVidia architecture. I thought this model might be appropriate due to the similarities in the problem domain of this simulator and their experiments.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model settled on a low mean squared error for both the training and validation set but while testing the model in the simulator the car tended to always drive one direction.

I attempted to use previous frames within the training system in order to provide a time series to the model for some time. Occasionally this had very promising results except for a few specific scenarios. In particular the model tended to attempt to drive towards water. This was probably due to the improper color channel differences between libraries. The training library was using BRG formatting while the testing library used RGB allowing the model to only use the green channel reliably.

After removing the time series information the issue with color channels was found and the system began to function except for two locations. the main issue was the transition between the bridge and the road. Using two laps of the difficult track caused the car to stay too close to the road edge due to the smaller road size on that track. As such, the training was changed to only include one lap of the hard track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architectureconsisted of a convolution neural network with an average pooling layer, five convolutional layers, six fully connected layers, dropout, and a flattening layer.

Here is a visualization of the architecture using keras' in built model visualizer:

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded five laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Then I repeated this process while driving track one backwards and two laps on track two in order to get more data points.

The data from all three cameras was used here. A +-0.6 angle was added or subtracted to the left and right images respectively. Here is an example of what a set of three images looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To further augment the data sat, I also flipped images and angles thinking that this would ensure an equal number of left and right frames. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 74844 number of data points using the forward data, backward data, and one lap on the hard track. I then preprocessed this data by calculating a norm of each image before feeding it into the network. To accomplish this, I used a keras lambda layer that divided each image by 128.5 and then subtracted 1 from each value.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the lack of any minimization on the validation set beyond that point. Also, as more epochs passed the model converged on always driving straight due to the fact that roughly 30% of the data outputed an angle of zero. Given that the validation accuracy was roughly 20% in the functional model and accuracy could hit 33% by finding the mean value of the labels any more epochs guaranteed that the model would fail. I used an adam optimizer so that manually training the learning rate wasn't necessary.
