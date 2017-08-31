#=================Imports==================#

#get the mnist data
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) #one hot means it labels the classes by one hot, so 1 for the correct class, 0 for the rest

import tensorflow as tf
sess = tf.InteractiveSession() #an interactive session connects to the backend to use the highly efficient C++ computations

#=================Tensorboard setup=================#

file_writer = tf.summary.FilerWriter('~/tensorflow/tutorials/logs', sess.graph)#setup the path for tensorflow logs

#=================Function Decleration=================#

#creates weight variables with some noise
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

#creates bias with a value slightly above 0
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#convolution with stride of 1 and zero padding
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME' )

#maxpooling with stride of 2 using a 2x2 filter
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 

#====================putting layers together====================#
#placeholders
x = tf.placeholder(tf.float32, shape=[None, 784]) #flattened input image
y_ = tf.placeholder(tf.float32, shape= [None, 10]) #labled classifications

#convolution and pooling 1
W_conv1 = weight_variable([5,5,1,32]) #5x5 filter, 1 input channel, 32 output channels
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1]) #28x28 input image with 1 color channel, the -1 just makes sure that the total size is kept constant during reshaping

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #convolution 1 (includes relu)
h_pool1 = max_pool_2x2(h_conv1) #maxpool 1

#convolution and pooling 2
W_conv2 = weight_variable([5,5,32,64]) #5x5 filter, 32 input channels, 64 output channels
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #convolution 2 (includes relu)
h_pool2 = max_pool_2x2(h_conv2) #maxpool 2

#fully connected
W_fc1 = weight_variable([7*7*64, 1024])#fully connected, 7x7x64 to 1024
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) #flatten the output of last layer so it is compatible for matrix multiplication
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32) #probability of keeping a neuron (not dropping it out)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #dropout some of the neurons to prevent overfitting

#readout layer (softmax!)
W_fc2 = weight_variable([1024,10]) #reduce the total outputs to 10 different classes
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#===================training====================#

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #learning rate of 1e-4 using ADAM optimizer
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) #check if the predicted class matches labled class
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(1000):
	batch  = mnist.train.next_batch(50) #load a batch from the training set
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={ x:batch[0], y_: batch[1], keep_prob: 1.0})
		print ("step %d, training accuracy %g" %(i, train_accuracy))
	
	train_step.run(feed_dict={x:batch[0], y_: batch[1], keep_prob: 0.5}) #run the training step using new values for x and y_

print ("test accuracy %g" %accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))












