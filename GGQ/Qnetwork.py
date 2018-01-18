import tensorflow as tf
class Qnetwork:
	def __init__(self):
		#These lines establish the feed-forward part of the network used to choose actions
		self.input_states = tf.placeholder(shape=[None,5],dtype=tf.float32,name="input1")

		#Layer 1
		# self.W1 = tf.Variable(tf.random_uniform([5,4],0,0.01))
		self.Qout1 = tf.layers.dense(inputs=self.input_states,units=4)

		# #Layer 2
		# #self.W2 = tf.Variable(tf.random_uniform([4,2],0,0.01))
		self.Qout2 = tf.layers.dense(inputs=self.Qout1,units=4)

		#Layer 3
		#self.W2 = tf.Variable(tf.random_uniform([4,2],0,0.01))
		self.Qout3 = tf.layers.dense(inputs=self.Qout1,units=2)

		self.predict = tf.argmax(self.Qout3,1,name="op_to_restore") # Remember to replace the Qout here with the most recent Qout

		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.y=tf.placeholder(shape=[None],dtype=tf.float32)
		self.online_q_values=self.Qout3
		self.actions_array=tf.placeholder(shape=[None],dtype=tf.int32)
		q_value=tf.reduce_sum(self.online_q_values*tf.one_hot(self.actions_array,2),axis=1,keep_dims=True)
		error=tf.abs(self.y-q_value)
		clipped_error = tf.clip_by_value(error, 0.0, 1.0)
		linear_error = 2 * (error - clipped_error)
		self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)
		self.trainer = tf.train.MomentumOptimizer(0.001,0.95, use_nesterov=True)
		self.updateModel = self.trainer.minimize(self.loss)

		#Tensorflow summary operations
		tf.summary.scalar("loss",self.loss)
		self.summary_op = tf.summary.merge_all()
