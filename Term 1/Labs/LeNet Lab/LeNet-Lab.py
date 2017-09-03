
# coding: utf-8

# # LeNet Lab
# ![LeNet Architecture](lenet.png)
# Source: Yan LeCun

# ## Load Data
# 
# Load the MNIST data, which comes pre-loaded with TensorFlow.
# 
# You do not need to modify this section.

# In[266]:

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


# The MNIST data that TensorFlow pre-loads comes as 28x28x1 images.
# 
# However, the LeNet architecture only accepts 32x32xC images, where C is the number of color channels.
# 
# In order to reformat the MNIST data into a shape that LeNet will accept, we pad the data with two rows of zeros on the top and bottom, and two columns of zeros on the left and right (28+2+2 = 32).
# 
# You do not need to modify this section.

# In[267]:

import numpy as np

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))


# ## Visualize Data
# 
# View a sample from the dataset.
# 
# You do not need to modify this section.

# In[268]:

import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])


# ## Preprocess Data
# 
# Shuffle the training data.
# 
# You do not need to modify this section.

# In[269]:

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)


# ## Setup TensorFlow
# The `EPOCH` and `BATCH_SIZE` values affect the training speed and model accuracy.
# 
# You do not need to modify this section.

# In[270]:

import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128


# ## TODO: Implement LeNet-5
# Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.
# 
# This is the only cell you need to edit.
# ### Input
# The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.
# 
# ### Architecture
# **Layer 1: Convolutional.** The output shape should be 28x28x6.
# 
# **Activation.** Your choice of activation function.
# 
# **Pooling.** The output shape should be 14x14x6.
# 
# **Layer 2: Convolutional.** The output shape should be 10x10x16.
# 
# **Activation.** Your choice of activation function.
# 
# **Pooling.** The output shape should be 5x5x16.
# 
# **Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.
# 
# **Layer 3: Fully Connected.** This should have 120 outputs.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 4: Fully Connected.** This should have 84 outputs.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 5: Fully Connected (Logits).** This should have 10 outputs.
# 
# ### Output
# Return the result of the 2nd fully connected layer.

# In[271]:

from tensorflow.contrib.layers import flatten

# Weights for Convolutional layer:
# tf.Variable(tf.truncated_normal(
#                [filter_size_width, filter_size_height, input_depth, k_output]))

# new_height = (input_height - filter_height + 2 * P)/S + 1
# new_width = (input_width - filter_width + 2 * P)/S + 1

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Store layers weight & bias
    weights = {
        'wc1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma)),
        'wc2': tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma)),
        'wd1': tf.Variable(tf.truncated_normal(shape=(5*5*16, 120), mean = mu, stddev = sigma)),
        'wd2': tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma)),
        'wd3': tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma))}

    biases = {
        'bc1': tf.Variable(tf.zeros([6])),
        'bc2': tf.Variable(tf.zeros([16])),
        'bd1': tf.Variable(tf.zeros([120])),
        'bd2': tf.Variable(tf.zeros([84])),
        'bd3': tf.Variable(tf.zeros([10]))}
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    c1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='VALID')
    c1 = tf.nn.bias_add(c1, biases['bc1'])
    print ("c1 output shape: {}. Desired: 28,28,6".format(c1.get_shape()))
    
    # TODO: Activation.
    a1 = tf.nn.relu(c1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    p1 = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print ("p1 output shape: {}. Desired: 14,14,6".format(p1.get_shape()))

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    c2 = tf.nn.conv2d(p1, weights['wc2'], strides=[1, 1, 1, 1], padding='VALID')
    c2 = tf.nn.bias_add(c2, biases['bc2'])
    print ("c2 output shape: {}. Desired: 10,10,16".format(c2.get_shape()))
    
    # TODO: Activation.
    a2 = tf.nn.relu(c2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    p2 = tf.nn.max_pool(a2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print ("p2 output shape: {}. Desired: 5,5,16".format(p2.get_shape()))

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    f = flatten(p2)
    print ("f output shape: {}. Desired: 400".format(f.get_shape()))
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = tf.reshape(f, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    print ("fc1 output shape: {}. Desired: 120".format(fc1.get_shape()))
    
    # TODO: Activation.
    a3 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.add(tf.matmul(a3, weights['wd2']), biases['bd2'])
    print ("fc2 output shape: {}. Desired: 84".format(fc2.get_shape()))
    
    # TODO: Activation.
    a4 = tf.nn.relu(fc2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3 = tf.add(tf.matmul(a4, weights['wd3']), biases['bd3'])
    print ("fc3 output shape: {}. Desired: 10".format(fc3.get_shape()))
    
    return fc3


# ## Features and Labels
# Train LeNet to classify [MNIST](http://yann.lecun.com/exdb/mnist/) data.
# 
# `x` is a placeholder for a batch of input images.
# `y` is a placeholder for a batch of output labels.
# 
# You do not need to modify this section.

# In[272]:

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)


# ## Training Pipeline
# Create a training pipeline that uses the model to classify MNIST data.
# 
# You do not need to modify this section.

# In[273]:

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# ## Model Evaluation
# Evaluate how well the loss and accuracy of the model for a given dataset.
# 
# You do not need to modify this section.

# In[274]:

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# ## Train the Model
# Run the training data through the training pipeline to train the model.
# 
# Before each epoch, shuffle the training set.
# 
# After each epoch, measure the loss and accuracy of the validation set.
# 
# Save the model after training.
# 
# You do not need to modify this section.

# In[275]:

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")


# ## Evaluate the Model
# Once you are completely satisfied with your model, evaluate the performance of the model on the test set.
# 
# Be sure to only do this once!
# 
# If you were to measure the performance of your trained model on the test set, then improve your model, and then measure the performance of your model on the test set again, that would invalidate your test results. You wouldn't get a true measure of how well your model would perform against real data.
# 
# You do not need to modify this section.

# In[276]:

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

