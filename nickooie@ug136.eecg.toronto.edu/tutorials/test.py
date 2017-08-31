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


print (sess.run(W))
print (sess.run(U))
sess.run(save_old)
sess.run(update)
print (sess.run(W))
print (sess.run(U))
sess.run(save_old)
sess.run(update)
print (sess.run(W))
print (sess.run(U))
