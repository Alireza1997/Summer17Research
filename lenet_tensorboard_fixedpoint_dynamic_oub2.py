#=================Imports==================#

#get the mnist data
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) #one hot means it labels the classes by one hot, so 1 for the correct class, 0 for the rest

import tensorflow as tf
import numpy as np

rnd = tf.load_op_library('fix_round_split_dynamic.so')

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient("FixRoundSplit")
def _fix_round_split_grad(op, grad):
  """The gradients for `zero_out`.

  Args:
    op: The `zero_out` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  return rnd.fix_round_split_grad(grad,op.inputs[0],op.inputs[1],op.inputs[2],op.inputs[3],op.inputs[4])


#=================Tensorboard setup=================#


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

def floatToFixPoint(tensor_):
	tensor_ = tf.clip_by_value(tensor_,-32.0,32.0)
	tensor_ = tf.scalar_mul(32768.0,tensor_)
	tensor_ = tf.round(tensor_)
	tensor_ = tf.scalar_mul(1/32768.0,tensor_)
	return (tensor_)

#====================putting layers together====================#
#placeholders
x = tf.placeholder(tf.float32, shape=[None, 784], name = 'input') #flattened input image
y_ = tf.placeholder(tf.float32, shape= [None, 10], name = 'labels') #labled classifications
y_conv = tf.Variable(tf.zeros([1,10]))
ILFLU = tf.Variable([5.,26.] ,trainable=False)
ILFLF = tf.Variable([5.,26.] ,trainable=False)
ILFLB = tf.Variable([5.,26.] ,trainable=False)
overflowW = tf.Variable([0.] ,trainable=False)
overflowB = tf.Variable([0.] ,trainable=False)
overflowA = tf.Variable([0.] ,trainable=False)
overflowG = tf.Variable([0.] ,trainable=False)
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
h_fc1 = rnd.fix_round_split(tf.nn.relu(tf.matmul(h_pool2_flat, rnd.fix_round_split(W_fc1,ILFLU,ILFLU,overflowW,overflowW)) + rnd.fix_round_split(b_fc1,ILFLU,ILFLU,overflowB,overflowB)),ILFLF,ILFLB,overflowA,overflowG)


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

#=================delta setup==================#

# W_old_c2 = tf.Variable(W_conv2.initialized_value(), name="old_W_conv2")
# save_old_c2 = tf.assign(W_old_c2,W_conv2)
# W_deltas_c2 = tf.Variable(tf.zeros([5,5,20,40]), name="deltas_c2")
# save_delta_c2 = tf.assign(W_deltas_c2, tf.subtract(W_old_c2,W_conv2))
# tf.summary.histogram("deltas_c2", W_deltas_c2)
# W_deltas_abs_c2 = tf.Variable(tf.zeros([5,5,20,40]), name="deltas_c2_abs")
# save_abs_delta_c2 = tf.assign(W_deltas_abs_c2, tf.abs(W_deltas_c2))
# tf.summary.histogram("deltas_abs_c2", W_deltas_abs_c2)
# tf.summary.scalar("deltas_abs_min_c2", tf.reduce_min(W_deltas_abs_c2))
# tf.summary.scalar("deltas_abs_max_c2", tf.reduce_max(W_deltas_abs_c2))
# tf.summary.scalar("deltas_mean_c2", tf.reduce_mean(W_deltas_abs_c2))

# W_old_c1 = tf.Variable(W_conv1.initialized_value(), name="old_W_conv1")
# save_old_c1 = tf.assign(W_old_c1,W_conv1)
# W_deltas_c1 = tf.Variable(tf.zeros([5,5,1,20]), name="deltas_c1")
# save_delta_c1 = tf.assign(W_deltas_c1, tf.subtract(W_old_c1,W_conv1))
# tf.summary.histogram("deltas_c1", W_deltas_c1)
# W_deltas_abs_c1 = tf.Variable(tf.zeros([5,5,1,20]), name="deltas_c1_abs")
# save_abs_delta_c1 = tf.assign(W_deltas_abs_c1, tf.abs(W_deltas_c1))
# tf.summary.histogram("deltas_abs_c1", W_deltas_abs_c1)
# tf.summary.scalar("deltas_abs_min_c1", tf.reduce_min(W_deltas_abs_c1))
# tf.summary.scalar("deltas_abs_max_c1", tf.reduce_max(W_deltas_abs_c1))
# tf.summary.scalar("deltas_mean_c1", tf.reduce_mean(W_deltas_abs_c1))

# W_old = tf.Variable(W_fc1.initialized_value(), name="old_W")
# save_old = tf.assign(W_old,W_fc1)
# W_deltas = tf.Variable(tf.zeros([4*4*40, 1000]), name="deltas_fc1")
# save_delta = tf.assign(W_deltas, tf.subtract(W_old,W_fc1))
# tf.summary.histogram("deltas_fc1", W_deltas)
# W_deltas_abs = tf.Variable(tf.zeros([4*4*40, 1000]), name="deltas_fc1_abs")
# save_abs_delta = tf.assign(W_deltas_abs, tf.abs(W_deltas))
# tf.summary.histogram("deltas_abs_fc1", W_deltas_abs)
# tf.summary.scalar("deltas_abs_min_fc1", tf.reduce_min(W_deltas_abs))
# tf.summary.scalar("deltas_abs_max_fc1", tf.reduce_max(W_deltas_abs))
# tf.summary.scalar("deltas_mean_fc1", tf.reduce_mean(W_deltas_abs))

# W_old2 = tf.Variable(W_fc2.initialized_value(), name="old_W2")
# save_old2 = tf.assign(W_old2,W_fc2)
# W_deltas2 = tf.Variable(tf.zeros([1000, 500]), name="deltas_fc2")
# save_delta2 = tf.assign(W_deltas2, tf.subtract(W_old2,W_fc2))
# tf.summary.histogram("deltas_fc2", W_deltas2)
# W_deltas2_abs = tf.Variable(tf.zeros([1000, 500]), name="deltas_fc2_abs")
# save_abs_delta2 = tf.assign(W_deltas2_abs, tf.abs(W_deltas2))
# tf.summary.histogram("deltas_abs_fc2", W_deltas2_abs)
# tf.summary.scalar("deltas_abs_min_fc2", tf.reduce_min(W_deltas2_abs))
# tf.summary.scalar("deltas_abs_max_fc2", tf.reduce_max(W_deltas2_abs))
# tf.summary.scalar("deltas_mean_fc2", tf.reduce_mean(W_deltas2_abs))

# W_old3 = tf.Variable(W_fc3.initialized_value(), name="old_W3")
# save_old3 = tf.assign(W_old3,W_fc3)
# W_deltas3 = tf.Variable(tf.zeros([500,10]), name="deltas_fc3")
# save_delta3 = tf.assign(W_deltas3, tf.subtract(W_old3,W_fc3))
# tf.summary.histogram("deltas_fc3", W_deltas3)
# W_deltas3_abs = tf.Variable(tf.zeros([500, 10]), name="deltas_fc3_abs")
# save_abs_delta3 = tf.assign(W_deltas3_abs, tf.abs(W_deltas3))
# tf.summary.histogram("deltas_abs_fc3", W_deltas3_abs)
# tf.summary.scalar("deltas_abs_min_fc3", tf.reduce_min(W_deltas3_abs))
# tf.summary.scalar("deltas_abs_max_fc3", tf.reduce_max(W_deltas3_abs))
# tf.summary.scalar("deltas_mean_fc3", tf.reduce_mean(W_deltas3_abs))

# W_init =  tf.Variable(W_fc2.initialized_value(), name="init_Wfc2")
# save_W_init = tf.assign(W_init,tf.subtract(W_init,W_fc2))

# tf.summary.histogram("W_c1", W_conv1)
# tf.summary.histogram("W_c2", W_conv2)
# tf.summary.histogram("W_fc1", W_fc1)
# tf.summary.histogram("W_fc2", W_fc2)
# tf.summary.histogram("W_fc3", W_fc3)


#===================training====================#

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #learning rate of 1e-4 using ADAM optimizer
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) #check if the predicted class matches labled class
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)


sess = tf.InteractiveSession() #an interactive session connects to the backend to use the highly efficient C++ computations
sess.run(tf.global_variables_initializer())

file_writer = tf.summary.FileWriter('/home/alireza/tensorflow/tutorials/logs', sess.graph)#setup the path for tensorflow logs
tf.summary.scalar("accuracy", accuracy)
merged = tf.summary.merge_all()

# dynamic set up
accuracy_peak = 0
avg_accuracy_old = 0
avg_accuracy = 0
sum_accuracy = 0
overflow_threshhold = 0
overflow_chunk = 10
accuracy_chunk = 10
setup_steps = 50

for i in range(1000):
	batch  = mnist.train.next_batch(50) #load a batch from the training set

	
	train_accuracy = accuracy.eval(feed_dict={ x:batch[0], y_: batch[1]})

	if (i > setup_steps):
		sum_accuracy += train_accuracy

	#manipulate ILFLs based on overflow
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

	if (i == setup_steps):
		sess.run(tf.assign(ILFLU,[0,15]))
		sess.run(tf.assign(ILFLB,[0,15]))
		sess.run(tf.assign(ILFLF,[0,15]))

	if (i % overflow_chunk and i > setup_steps) == 0:
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

			sess.run(tf.assign(overflowW, [0]))
			sess.run(tf.assign(overflowB, [0]))
		if (og[0] > overflow_threshhold):
			print("ILFLB: \t"),	
			if (sess.run(ILFLB)[1] > 0):
				print sess.run(tf.assign(ILFLB,tf.add(ILFLB,[1,-1])))
			else:
				print sess.run(tf.assign(ILFLB,tf.add(ILFLB,[1,0])))
			
			sess.run(tf.assign(overflowG, [0]))
		if (oa[0] > overflow_threshhold):
			print("ILFLF: \t"),	
			if (sess.run(ILFLF)[1] > 0):
				print sess.run(tf.assign(ILFLF,tf.add(ILFLF,[1,-1])))
			else:
				print sess.run(tf.assign(ILFLF,tf.add(ILFLF,[1,0])))

			sess.run(tf.assign(overflowA, [0]))

	#manipulate ILFLs based on accuracy
	if (i% accuracy_chunk == 0 and  i > setup_steps):
		if (avg_accuracy_old > accuracy_peak):
			accuracy_peak = avg_accuracy_old

		avg_accuracy_old = avg_accuracy
		avg_accuracy = sum_accuracy/accuracy_chunk
		sum_accuracy = 0

		print(avg_accuracy)

		if (avg_accuracy < accuracy_peak*0.9):
			ilflu = sess.run(ILFLU)
			ilflf = sess.run(ILFLF)
			ilflb = sess.run(ILFLB)
			
			if (ilflu[1] < 25):
				ilflu[1]+=1

			if (ilflf[1] < 25):
				ilflf[1]+=1

			if (ilflb[1] < 25):
				ilflb[1]+=1
			
			print("ILFLU: \t"),	
			print (sess.run(tf.assign(ILFLU,ilflu)))
			print("ILFLF: \t"),	
			print (sess.run(tf.assign(ILFLF,ilflf)))
			print("ILFLB: \t"),	
			print (sess.run(tf.assign(ILFLB,ilflb)))

		

	
	# sess.run(save_old_c1)
	# sess.run(save_old_c2)
	# sess.run(save_old) #save W before it changes
	# sess.run(save_old2)
	# sess.run(save_old3)
	#print sess.run(W_conv2)
	train_step.run(feed_dict={x:batch[0], y_: batch[1]}) #run the training step using new values for x and y_, update weights
	# sess.run(save_delta_c1)
	# sess.run(save_abs_delta_c1)
	# sess.run(save_delta_c2)
	# sess.run(save_abs_delta_c2)
	# sess.run(save_delta) #save the delta value after weights change
	# sess.run(save_abs_delta)
	# sess.run(save_delta2)
	# sess.run(save_abs_delta2)
	# sess.run(save_delta3)
	# sess.run(save_abs_delta3)

	# if i==333:
	# 	sess.run(tf.assign(ILFLU,[1.,15.]))
	# 	sess.run(tf.assign(ILFLF,[5.,11.]))
	# 	sess.run(tf.assign(ILFLB,[1.,15.]))

	# if i==666:
	# 	sess.run(tf.assign(ILFLU,[5.,15.]))
	# 	sess.run(tf.assign(ILFLF,[5.,15.]))
	# 	sess.run(tf.assign(ILFLB,[5.,15.]))

	result = sess.run(merged, feed_dict={ x:batch[0], y_: batch[1]})	
	file_writer.add_summary(result, i)	
		#print (sess.run(tf.nn.softmax(y_conv), feed_dict={ x:batch[0], y_:batch[1]}))
		#print (batch[1])
		
	if i%100 == 0:
		print sess.run(W_fc2)
		# print sess.run(tf.subtract(W_init,W_fc2))
		#print (sess.run(W_fc1)) #print out the deltas every 100 steps
		#print (sess.run(W_deltas_abs))		
	
print ("test accuracy %g" %accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels}))
print sess.run(W_fc2)
# print(sess.run(save_W_init))
# print(sess.run(tf.divide(W_init,2000)))











