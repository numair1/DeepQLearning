import tensorflow as tf
class Qnetwork:
	def __init__(self):
		#These lines establish the feed-forward part of the network used to choose actions
		self.input_states = tf.placeholder(shape=[None,5],dtype=tf.float32,name="input1")

		#Layer 1
		# self.W1 = tf.Variable(tf.random_uniform([5,4],0,0.01))
		self.Qout1 = tf.layers.dense(inputs=self.input_states,units=4)

		#Layer 2
		#self.W2 = tf.Variable(tf.random_uniform([4,2],0,0.01))
		self.Qout2 = tf.layers.dense(inputs=self.Qout1,units=4,activation=tf.nn.relu)

		#Layer 3
		#self.W2 = tf.Variable(tf.random_uniform([4,2],0,0.01))
		self.Qout3 = tf.layers.dense(inputs=self.Qout2,units=2)

		#Layer3
		# self.W3 = tf.Variable(tf.random_uniform([4,2],0,0.01))
		# self.Qout3 = tf.matmul(self.Qout2,self.W3)

		self.predict = tf.argmax(self.Qout3,1,name="op_to_restore") # Remember to replace the Qout here with the most recent Qout
		self.nextQ = tf.placeholder(shape=[None,2],dtype=tf.float32)
		self.square=tf.square(self.nextQ - self.Qout3)
		self.loss = tf.reduce_mean(tf.square(self.nextQ - self.Qout3))# Remember to replace the Qout here with the most recent Qout
		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
		self.updateModel = self.trainer.minimize(self.loss)

		#Tensorflow summary operations
		tf.summary.scalar("loss",self.loss)
		self.summary_op = tf.summary.merge_all()
