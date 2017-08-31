#=================Imports==================#

#get the mnist data
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) #one hot means it labels the classes by one hot, so 1 for the correct class, 0 for the rest

import tensorflow as tf
sess = tf.InteractiveSession() #an interactive session connects to the backend to use the highly efficient C++ computations

#=================Tensorboard setup=================#

file_writer = tf.summary.FileWriter('/home/alireza/tensorflow/tutorials/logs', sess.graph)#setup the path for tensorflow logs

#=================Function Decleration=================#

#creates weight variables with some noise
def weight_variable(shape, _name):
	initial = tf.truncated_normal(shape, stddev = 0.1, name = _name)
	return tf.Variable(initial)

#creates bias with a value slightly above 0
def bias_variable(shape, _name):
	initial = tf.constant(0.1, shape=shape, name = _name)
	return tf.Variable(initial)

#convolution with stride of 1 and no padding
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')

#maxpooling with stride of 2 using a 2x2 filter and no padding
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') 

#====================putting layers together====================#
#placeholders
x = tf.placeholder(tf.float32, shape=[None, 784], name = 'input') #flattened input image
y_ = tf.placeholder(tf.float32, shape= [None, 10], name = 'labels') #labled classifications

#convolution and pooling 1
with tf.name_scope("ConvPool1"):
	with tf.name_scope("Weights"):
		W_conv1 = weight_variable([5,5,1,20], 'W1') #5x5 filter, 1 input channel, 20 output channels
	with tf.name_scope("biases"):
		b_conv1 = bias_variable([20],'B1')
	
	with tf.name_scope("input"):
		x_image = tf.reshape(x, [-1,28,28,1]) #28x28 input image with 1 color channel, the -1 just makes sure that the total size is kept constant during reshaping

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #convolution 1 (includes relu)
	h_pool1 = max_pool_2x2(h_conv1) #maxpool 1

#convolution and pooling 2
W_conv2 = weight_variable([5,5,20,40],'W2') #5x5 filter, 20 input channels, 40 output channels
b_conv2 = bias_variable([40],'B2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #convolution 2 (includes relu)
h_pool2 = max_pool_2x2(h_conv2) #maxpool 2

#fully connected
W_fc1 = weight_variable([4*4*40, 1000],'W3')#fully connected, 4x4x40 to 6400
b_fc1 = bias_variable([1000],'B3')

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*40]) #flatten the output of last layer so it is compatible for matrix multiplication
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


W_fc2 = weight_variable([1000, 500], 'W4')#fully connected, 4x4x40 to 6400
b_fc2 = bias_variable([500], 'B4')

h_fc1_flat = tf.reshape(h_fc1, [-1, 1000]) #flatten the output of last layer so it is compatible for matrix multiplication
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_flat, W_fc2) + b_fc2)

#dropout
#keep_prob = tf.placeholder(tf.float32) #probability of keeping a neuron (not dropping it out)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #dropout some of the neurons to prevent overfitting

#readout layer (softmax!)
W_fc3 = weight_variable([500,10],'W5') #reduce the total outputs to 10 different classes
b_fc3 = bias_variable([10],'B5')

y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

#=================delta setup==================#


W_old_c2 = tf.Variable(W_conv2.initialized_value(), name="old_W_conv2")
save_old_c2 = tf.assign(W_old_c2,W_conv2)
W_deltas_c2 = tf.Variable(tf.zeros([5,5,20,40]), name="deltas_c2")
save_delta_c2 = tf.assign(W_deltas_c2, tf.subtract(W_old_c2,W_conv2))
tf.summary.histogram("deltas_c2", W_deltas_c2)
W_deltas_abs_c2 = tf.Variable(tf.zeros([5,5,20,40]), name="deltas_c2_abs")
save_abs_delta_c2 = tf.assign(W_deltas_abs_c2, tf.abs(W_deltas_c2))
tf.summary.histogram("deltas_abs_c2", W_deltas_abs_c2)
tf.summary.scalar("deltas_abs_min_c2", tf.reduce_min(W_deltas_abs_c2))
tf.summary.scalar("deltas_abs_max_c2", tf.reduce_max(W_deltas_abs_c2))

W_old_c1 = tf.Variable(W_conv1.initialized_value(), name="old_W_conv1")
save_old_c1 = tf.assign(W_old_c1,W_conv1)
W_deltas_c1 = tf.Variable(tf.zeros([5,5,1,20]), name="deltas_c1")
save_delta_c1 = tf.assign(W_deltas_c1, tf.subtract(W_old_c1,W_conv1))
tf.summary.histogram("deltas_c1", W_deltas_c1)
W_deltas_abs_c1 = tf.Variable(tf.zeros([5,5,1,20]), name="deltas_c1_abs")
save_abs_delta_c1 = tf.assign(W_deltas_abs_c1, tf.abs(W_deltas_c1))
tf.summary.histogram("deltas_abs_c1", W_deltas_abs_c1)
tf.summary.scalar("deltas_abs_min_c1", tf.reduce_min(W_deltas_abs_c1))
tf.summary.scalar("deltas_abs_max_c1", tf.reduce_max(W_deltas_abs_c1))

W_old = tf.Variable(W_fc1.initialized_value(), name="old_W")
save_old = tf.assign(W_old,W_fc1)
W_deltas = tf.Variable(tf.zeros([4*4*40, 1000]), name="deltas_fc1")
save_delta = tf.assign(W_deltas, tf.subtract(W_old,W_fc1))
tf.summary.histogram("deltas_fc1", W_deltas)
W_deltas_abs = tf.Variable(tf.zeros([4*4*40, 1000]), name="deltas_fc1_abs")
save_abs_delta = tf.assign(W_deltas_abs, tf.abs(W_deltas))
tf.summary.histogram("deltas_abs_fc1", W_deltas_abs)
tf.summary.scalar("deltas_abs_min_fc1", tf.reduce_min(W_deltas_abs))
tf.summary.scalar("deltas_abs_max_fc1", tf.reduce_max(W_deltas_abs))

W_old2 = tf.Variable(W_fc2.initialized_value(), name="old_W2")
save_old2 = tf.assign(W_old2,W_fc2)
W_deltas2 = tf.Variable(tf.zeros([1000, 500]), name="deltas_fc2")
save_delta2 = tf.assign(W_deltas2, tf.subtract(W_old2,W_fc2))
tf.summary.histogram("deltas_fc2", W_deltas2)
W_deltas2_abs = tf.Variable(tf.zeros([1000, 500]), name="deltas_fc2_abs")
save_abs_delta2 = tf.assign(W_deltas2_abs, tf.abs(W_deltas2))
tf.summary.histogram("deltas_abs_fc2", W_deltas2_abs)
tf.summary.scalar("deltas_abs_min_fc2", tf.reduce_min(W_deltas2_abs))
tf.summary.scalar("deltas_abs_max_fc2", tf.reduce_max(W_deltas2_abs))


tf.summary.histogram("W_c1", W_conv1)
tf.summary.histogram("W_c2", W_conv2)
tf.summary.histogram("W_fc1", W_fc1)
tf.summary.histogram("W_fc2", W_fc2)

merged = tf.summary.merge_all()

#===================training====================#

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #learning rate of 1e-4 using ADAM optimizer
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) #check if the predicted class matches labled class
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(2000):
	batch  = mnist.train.next_batch(50) #load a batch from the training set

	if i%10 == 0:
		train_accuracy = accuracy.eval(feed_dict={ x:batch[0], y_: batch[1]})
		print ("step %d, training accuracy %g" %(i, train_accuracy))
	
	sess.run(save_old_c1)
	sess.run(save_old_c2)
	sess.run(save_old) #save W before it changes
	sess.run(save_old2)
	train_step.run(feed_dict={x:batch[0], y_: batch[1]}) #run the training step using new values for x and y_, update weights
	sess.run(save_delta_c1)
	sess.run(save_abs_delta_c1)
	sess.run(save_delta_c2)
	sess.run(save_abs_delta_c2)
	sess.run(save_delta) #save the delta value after weights change
	sess.run(save_abs_delta)
	sess.run(save_delta2)
	sess.run(save_abs_delta2)


	if i < 50:
		result = sess.run(merged, feed_dict={ x:batch[0], y_: batch[1]})	
		file_writer.add_summary(result, i)	
		print (sess.run(tf.reduce_min(W_deltas_abs)))
		
	if i%100 == 0:
		print (sess.run(W_deltas)) #print out the deltas every 100 steps
		print (sess.run(W_deltas_abs))		
	
print ("test accuracy %g" %accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels}))












