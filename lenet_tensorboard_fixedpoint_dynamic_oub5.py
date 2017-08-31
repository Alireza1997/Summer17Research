#=================Imports==================#

#get the mnist data
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) #one hot means it labels the classes by one hot, so 1 for the correct class, 0 for the rest

import tensorflow as tf
import numpy as np

rnd = tf.load_op_library('fix_round_split_dynamic2.so')

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

#gradient registration
@ops.RegisterGradient("FixRoundSplit")
def _fix_round_split_grad(op, grad):
  
  return rnd.fix_round_split_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],op.inputs[3],op.inputs[4])

#=================Function Decleration=================#

#creates weight variables with some noise
def weight_variable(shape, _name):
	initial = tf.truncated_normal(shape, stddev=0.1, name = _name)
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
y_conv = tf.Variable(tf.zeros([1,10]))

#initializing precision variables and overflow variables
ILFLU = tf.Variable([0.,32.] ,trainable=False)
ILFLF = tf.Variable([0.,32.] ,trainable=False)
ILFLB = tf.Variable([0.,32.] ,trainable=False)
overflowW = tf.Variable([0.,0] ,trainable=False)
overflowB = tf.Variable([0.,0] ,trainable=False)
overflowA = tf.Variable([0.,0] ,trainable=False)
overflowG = tf.Variable([0.,0] ,trainable=False)

#plots for tensorboard
tf.summary.scalar("ILFLU1", ILFLU[0])
tf.summary.scalar("ILFLU2", ILFLU[1])
tf.summary.scalar("ILFLF1", ILFLF[0])
tf.summary.scalar("ILFLF2", ILFLF[1])
tf.summary.scalar("ILFLB1", ILFLB[0])
tf.summary.scalar("ILFLB2", ILFLB[1])

#convolution and pooling 1
with tf.name_scope("ConvPool1"):
	with tf.name_scope("Weights"):
		W_conv1 = weight_variable([5,5,1,20], 'W1') #5x5 filter, 1 input channel, 20 output channels
	with tf.name_scope("biases"):
		b_conv1 = bias_variable([20],'B1')
	
	with tf.name_scope("input"):
		x_image = rnd.fix_round_split(tf.reshape(x, [-1,28,28,1]),ILFLF,ILFLB,overflowA,overflowG) #28x28 input image with 1 color channel, the -1 just makes sure that the total size is kept constant during reshaping

	h_conv1 = tf.nn.relu(conv2d(x_image, rnd.fix_round_split(W_conv1,ILFLU,ILFLU,overflowW,overflowW)) + rnd.fix_round_split(b_conv1,ILFLU,ILFLU,overflowB,overflowB)) #convolution 1 (includes relu)
	h_pool1 = rnd.fix_round_split(max_pool_2x2(h_conv1),ILFLF,ILFLB,overflowA,overflowG) #maxpool 1

#convolution and pooling 2
W_conv2 = weight_variable([5,5,20,40],'W2') #5x5 filter, 20 input channels, 40 output channels
b_conv2 = bias_variable([40],'B2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, rnd.fix_round_split(W_conv2,ILFLU,ILFLU,overflowW,overflowW)) + rnd.fix_round_split(b_conv2,ILFLU,ILFLU,overflowB,overflowB)) #convolution 2 (includes relu)
h_pool2 = rnd.fix_round_split((max_pool_2x2(h_conv2)),ILFLF,ILFLB,overflowA,overflowG) #maxpool 2

#fully connected
W_fc1 = weight_variable([4*4*40, 1000],'W3')#fully connected, 4x4x40 to 6400
b_fc1 = bias_variable([1000],'B3')

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*40]) #flatten the output of last layer so it is compatible for matrix multiplication
h_fc1 = rnd.fix_round_split(tf.nn.relu(tf.matmul(h_pool2_flat, rnd.fix_round_split(W_fc1,ILFLU,ILFLU,overflowW,overflowW))
 + rnd.fix_round_split(b_fc1,ILFLU,ILFLU,overflowB,overflowB)),ILFLF,ILFLB,overflowA,overflowG)


W_fc2 = weight_variable([1000, 500], 'W4')#fully connected, 4x4x40 to 6400
b_fc2 = bias_variable([500], 'B4')

h_fc1_flat = tf.reshape(h_fc1, [-1, 1000]) #flatten the output of last layer so it is compatible for matrix multiplication
h_fc2 = rnd.fix_round_split(tf.nn.relu(tf.matmul(h_fc1_flat, rnd.fix_round_split(W_fc2,ILFLU,ILFLU,overflowW,overflowW)) + rnd.fix_round_split(b_fc2,ILFLU,ILFLU,overflowB,overflowB)),ILFLF,ILFLB,overflowA,overflowG)

#dropout
#keep_prob = tf.placeholder(tf.float32) #probability of keeping a neuron (not dropping it out)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #dropout some of the neurons to prevent overfitting

#readout layer (softmax!)
W_fc3 = weight_variable([500,10],'W5') #reduce the total outputs to 10 different classes
b_fc3 = bias_variable([10],'B5')

y_conv = rnd.fix_round_split((tf.matmul(h_fc2, rnd.fix_round_split(W_fc3,ILFLU,ILFLU,overflowW,overflowW)) + rnd.fix_round_split(b_fc3,ILFLU,ILFLU,overflowB,overflowB)),ILFLF,ILFLB,overflowA,overflowG)

#===================training====================#

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #learning rate of 1e-4 using ADAM optimizer
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) #check if the predicted class matches labled class
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)


sess = tf.InteractiveSession() #an interactive session connects to the backend to use the highly efficient C++ computations
sess.run(tf.global_variables_initializer())

file_writer = tf.summary.FileWriter('/home/alireza/tensorflow/tutorials/logs', sess.graph)#setup the path for tensorflow logs
merged = tf.summary.merge_all()

# dynamic precision variable set up
accuracy_peak = 0				
avg_accuracy_old = 0
avg_accuracy = 0
sum_accuracy = 0
overflow_threshhold = 0				#How much overflow is allowed before the range bitwidth is increased
overflow_chunk = 1					#over how many epochs should it check for overflow
accuracy_chunk = 10					#over how many epochs should it grab the average accuracy
setup_steps = 50					#how many steps at the start of training should be done at full precision
target_length = 10					#to what bitwidth size should the precision be reduced to after the setup steps are over
final_steps = 100						#how many steps at the end of training should be done at full precision
epochs = 1000						#how many epochs should the model train for
final_length = 30					#the bit width used during the 'final_steps'
accuracy_tolerance = 0.97			#the required amount of accuracy, if not met the model will increase its precision
underflow_tolerance = 0.05			#the threshold for the ammount of underflows, if exceeded the model will increase its precision


#training loop
for i in range(epochs):
	batch  = mnist.train.next_batch(50) #load a batch from the training set

	
	train_accuracy = accuracy.eval(feed_dict={ x:batch[0], y_: batch[1]})

	if (i > setup_steps):
		sum_accuracy += train_accuracy

	if (i % 10) == 0:
		print ("step %d, training accuracy %g" %(i, train_accuracy))
		print("weights overflow: \t"),	
		print(sess.run(overflowW))
		print("biases overflow: \t"),
		print(sess.run(overflowB))
		print("activations overflow: \t"),
		print(sess.run(overflowA))
		print("gradients overflow: \t"),
		print(sess.run(overflowG))
		print("ILFLU: \t"),	
		print (sess.run(ILFLU))
		print("ILFLF: \t"),	
		print (sess.run(ILFLF))
		print("ILFLB: \t"),	
		print (sess.run(ILFLB))
		print(accuracy_peak)
		print(avg_accuracy)

	#increase IL until there is no overflow
	if (i < setup_steps):

		if (i % overflow_chunk) == 0:
			ow = sess.run(overflowW)
			ob = sess.run(overflowB)
			oa = sess.run(overflowA)
			og = sess.run(overflowG)

			if ((ow[0] > overflow_threshhold) or (ob[0] > overflow_threshhold)):
				print("ILFLU: \t"),	
				if (sess.run(ILFLU)[1] > 0):
					print sess.run(tf.assign(ILFLU,tf.add(ILFLU,[1,-1])))
				else:
					print sess.run(tf.assign(ILFLU,tf.add(ILFLU,[1,0])))

				sess.run(tf.assign(overflowW, [0,ow[1]]))
				sess.run(tf.assign(overflowB, [0,ob[1]]))

			if (og[0] > overflow_threshhold):
				print("ILFLB: \t"),	
				if (sess.run(ILFLB)[1] > 0):
					print sess.run(tf.assign(ILFLB,tf.add(ILFLB,[1,-1])))
				else:
					print sess.run(tf.assign(ILFLB,tf.add(ILFLB,[1,0])))
				
				sess.run(tf.assign(overflowG, [0,og[1]]))

			if (oa[0] > overflow_threshhold):
				print("ILFLF: \t"),	
				if (sess.run(ILFLF)[1] > 0):
					print sess.run(tf.assign(ILFLF,tf.add(ILFLF,[1,-1])))
				else:
					print sess.run(tf.assign(ILFLF,tf.add(ILFLF,[1,0])))

				sess.run(tf.assign(overflowA, [0, oa[1]]))

	#wait until training is "stabalized" then reduce the precision to target length
	elif (i == setup_steps):
		ilflu = sess.run(ILFLU)
		ilflf = sess.run(ILFLF)
		ilflb = sess.run(ILFLB)

		sess.run(tf.assign(ILFLU,[ilflu[0],target_length-ilflu[0]]))
		sess.run(tf.assign(ILFLB,[ilflb[0],target_length-ilflb[0]]))
		sess.run(tf.assign(ILFLF,[ilflf[0],target_length-ilflf[0]]))

	#increase the precision of the variable catagory that exceeded the underflow threshold
	elif (i < epochs-final_steps):
		if (i% accuracy_chunk == 0):
			if (avg_accuracy_old > accuracy_peak):
				accuracy_peak = avg_accuracy_old

			avg_accuracy_old = avg_accuracy
			avg_accuracy = sum_accuracy/accuracy_chunk
			sum_accuracy = 0

			ow = sess.run(overflowW)
			ob = sess.run(overflowB)
			oa = sess.run(overflowA)
			og = sess.run(overflowG)

			if (ow[1] > underflow_tolerance or ob[1] > underflow_tolerance):
				ilflu = sess.run(ILFLU)
				ilflu[1]+=1
				print("ILFLU: \t"),	
				print (sess.run(tf.assign(ILFLU,ilflu)))

			if (oa[1] > underflow_tolerance):	
				ilflf = sess.run(ILFLF)
				ilflf[1]+=1
				print("ILFLF: \t"),	
				print (sess.run(tf.assign(ILFLF,ilflf)))
				
			if (og[1] > underflow_tolerance):				
				ilflb = sess.run(ILFLB)
				ilflb[1]+=1				
				print("ILFLB: \t"),	
				print (sess.run(tf.assign(ILFLB,ilflb)))

	#change precision to final_length for the final_steps of training
	elif (i == epochs-final_steps):
		ilflu = sess.run(ILFLU)
		ilflf = sess.run(ILFLF)
		ilflb = sess.run(ILFLB)

		sess.run(tf.assign(ILFLU,[ilflu[0],final_length-ilflu[0]]))
		sess.run(tf.assign(ILFLB,[ilflb[0],final_length-ilflb[0]]))
		sess.run(tf.assign(ILFLF,[ilflf[0],final_length-ilflf[0]]))

		if (i% accuracy_chunk == 0):
			if (avg_accuracy_old > accuracy_peak):
				accuracy_peak = avg_accuracy_old

			avg_accuracy_old = avg_accuracy
			avg_accuracy = sum_accuracy/accuracy_chunk
			sum_accuracy = 0

	#run training at final_length precision for the final_steps of training
	else:
		if (i% accuracy_chunk == 0):
			if (avg_accuracy_old > accuracy_peak):
				accuracy_peak = avg_accuracy_old

			avg_accuracy_old = avg_accuracy
			avg_accuracy = sum_accuracy/accuracy_chunk
			sum_accuracy = 0

	
	train_step.run(feed_dict={x:batch[0], y_: batch[1]}) #run the training step using new values for x and y_, update weights


	result = sess.run(merged, feed_dict={ x:batch[0], y_: batch[1]})	
	file_writer.add_summary(result, i)	

print ("test accuracy %g" %accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels}))












