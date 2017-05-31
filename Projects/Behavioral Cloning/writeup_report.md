# Behavioral Cloning
## by Soeren Walls

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: examples/center_driving.jpg "Center Lane Driving Image"
[image2]: examples/recovery_1.jpg "Recovery Image 1"
[image3]: examples/recovery_2.jpg "Recovery Image 2"
[image4]: examples/recovery_3.jpg "Recovery Image 3"
[image5]: examples/normal.jpg "Normal Image"
[image6]: examples/flipped.jpg "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

1\. **Submission includes all required files and can be used to run the simulator in autonomous mode.**

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

2\. **Submission includes functional code**

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

3\. **Submission code is usable and readable**

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

1\. **An appropriate model arcthiecture has been employed**

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 81-90) 

The model includes RELU activation layers to introduce nonlinearity (code lines 83, 86, 89, 92, 95, 97), and the data is normalized in the model using a Keras lambda layer (code line 79). Max pooling with a 2x2 kernel is used after each convolutional layer to reduce dimensionality (code lines 82, 85, 88), and the input is cropped so that irrelevant information is ignored (code line 80). The model concludes with dense layers.

2\. **Attempts to reduce overfitting in the model**

The model contains a dropout layer in order to reduce overfitting (model.py line 91). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 31-38). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

3\. **Model parameter tuning**

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 101).

4\. **Appropriate training data**

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving in reverse, and driving on the second, more challenging, track. Ultimately, however, I ended up only using center lane driving data to produce my model, since the inclusion of the other data actually made the model less reliable on the first track.

In all of my data, I only included images from the center camera, since this was simpler than including the left and right cameras and adjusting the corresponding angles.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

1\. **Solution Design Approach**

The overall strategy for deriving a model architecture was to augment the NVIDIA architecture.

My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because it has been proven to work for self-driving car technology in the past, and it seems slightly more sophisticated than the LeNet model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding a dropout layer, and removing one of the convolution layers.

Then I decided to add a cropping layer to remove irrelevant information from the images like the trees and the sky, which reduced the time it took to fit the model to the training data, and also slightly improved the accuracy on the validation set.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track; usually it was just before the bridge. To improve the driving behavior in these cases, I decided to collect some of my own driving data to augment the provided data. At first, I collected more center lane driving data (code line 32) by manually driving around track one a couple of times. It turns out that this alone was enough extra data to create a good model that successfully drove the car around track one autonomously without falling off the track.

Despite having made a successful model, I noticed it was a bit wobbly at points, and I wanted to see if adding more data would improve this. I manually created more training data of the car driving in reverse, recovering from the sides of the road, and even center lane driving on track two. This certainly made my model more generalizable, since it was now able to drive for longer on track two without falling off, but it ultimately made the model perform worse on track one. I could not get it to stay on the track when using all of this extra training data, so I decided to revert to the model with just the center lane driving data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

2\. **Final Model Architecture**

The final model architecture (model.py lines 78-99) consisted of a convolution neural network with the following layers and layer sizes:

	1. Lambda (160x320x3)
	2. Cropping2D (70x25x3)
	3. Convolution2D (24@70x25) with 5x5 kernel
	4. MaxPooling2D (24x35x12x3)
	5. Activation (RELU)
	6. Convolution2D (36@35x12) with 5x5 kernel
	7. MaxPooling2D (36x17x6x3)
	8. Activation (RELU)
	9. Convolution2D (48@17x6) with 5x5 kernel
	10. MaxPooling2D (48x8x3x3)
	11. Activation (RELU)
	12. Convolution2D (64@8x3) with 3x3 kernel
	13. MaxPooling2D (64x4x1x3)
	14. Activation (RELU)
	15. Dropout (32x4x1x3)
	16. Activation (RELU)
	17. Flatten (384)
	18. Dense (100)
	19. Activation (RELU)
	20. Dense (50)
	21. Activation (RELU)
	22. Dense (10)
	23. Dense (1)

3\. **Creation of the Training Set & Training Process**

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself if it got off track. These images show what a recovery looks like starting from the right side of the track:

![alt text][image2]
![alt text][image3]
![alt text][image4]

I also recorded the vehicle driving in the reverse direction around track one, with the intention of generalizing the model even further.

Then I repeated this process on track two in order to get more data points. As previously mentioned, however, in the final model, I decided not to include the data from track two, recovery driving, or reverse driving, since it overcomplicated the model.

To augment the data sat, I also flipped images and angles thinking that this would help generalize the training data, prevent overfitting, and essentially double the amount of data collected. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

I collected a total of 45,528 images, and after flipping them, I had a total of 91,056 data points. However, I only used 63,036 data points, since I decided to exclude some of the data for reasons I explained previously. I then preprocessed this data by cropping it to exclude irrelevant image information like the sky and trees. Cropping is actually performed in the model, with a Cropping2D layer, since this is more efficient.

I finally randomly shuffled the data set and put 20% of the data into a validation set. I used a generator to feed the data into the model, rather than storing all of it in memory at once. This decreased memory usage and time needed to train the model.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the fact that 3 or more epochs resulted in overfitting (increased training accuracy and decreased validation accuracy). I used an adam optimizer so that manually training the learning rate wasn't necessary.
