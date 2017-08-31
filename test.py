import tensorflow as tf

W = tf.Variable(0,  name='weight')

U = tf.Variable(W.initialized_value(), name='old_weight')

one = tf.constant(1)

update = tf.assign(W,tf.add(W,one))
save_old = tf.assign(U,W)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


#print (sess.run(W))
#print (sess.run(U))
#sess.run(update)
#print (sess.run(W))
#print (sess.run(U))
#sess.run(update)
#print (sess.run(W))
#print (sess.run(U))


# print (sess.run(W))
# print (sess.run(U))
# sess.run(save_old)
# sess.run(update)
# print (sess.run(W))
# print (sess.run(U))
# sess.run(save_old)
# sess.run(update)
# print (sess.run(W))
# print (sess.run(U))




# v = tf.Variable(40.0)
# c = tf.constant(2.0)
# v = tf.cond(v < c, lambda: tf.constant(3.0), lambda:  v)
# # calc = tf.assign(v, c2)
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	# sess.run(calc)
# 	print sess.run(v)

# def floatToFixPoint(tensor_):
# 	c = tf.constant(32.0)
# 	tensor_ = tf.cond(tensor_ > c, lambda: c, lambda: c)
# 	tensor_ = tf.cond(tensor_ < -c, lambda: -c, lambda: c)
# 	return (tensor_)


# with tf.Session() as sess:
# 	v = floatToFixPoint(v)

# 	sess.run(tf.global_variables_initializer())
# 	print sess.run(v)

w = tf.Variable([ 2.11111111])
rnd = tf.load_op_library('fix_round_split_dynamic2.so')
overflow = tf.Variable([0.,0.],trainable=False)
boverflow = tf.Variable([0.,0.],trainable=False)
b = rnd.fix_round_split(w,[5,1],[5,1],overflow, boverflow)
# with tf.Session(''):
  # rnd.fix_round([[100., 2.], [3.11111111111, 4.]]).eval()

#config = tf.ConfigProto(log_device_placement = True)
#config.graph_options.optimizer_options.opt_level = -1

with tf.Session() as sess:
 	sess.run(tf.global_variables_initializer())
 	print (sess.run(b))
 	print (sess.run(overflow))
 	print (sess.run(boverflow))
# 	print sess.run(w)
# 	w = tf.clip_by_value(w,-32.0,32.0)
# 	print sess.run(w)
# 	w = tf.scalar_mul(256.0,w)
# 	print sess.run(w)
# 	w = tf.round(w)
# 	print sess.run(w)
# 	w = tf.scalar_mul(1/256.0,w)
# 	print sess.run(w)