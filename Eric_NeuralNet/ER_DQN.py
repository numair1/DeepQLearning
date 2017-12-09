import numpy as np
import random
import tensorflow as tf
import math
import json

class FOMS:

    def get_reward(self,state,action):
        normal_coeff_p1=2*(state[0]+state[1])-(state[2]+state[3])
        normal_coeff_p2=2*(state[2]+state[3])-(state[0]+state[1])
        return np.random.normal(loc=(1-action)*(normal_coeff_p1)+action*(normal_coeff_p2),scale=0.1,size=1)[0]

    def get_next_state(self,current_state,action):
        if len(current_state)==0:
            return np.random.normal(loc=0,scale=0.5,size=64)
        else:
            new_state=np.zeros(64)
            for i in range(1,17):
                s_t=current_state[i-1]
                model_s_t=s_t

                #mean and sd for normal distribution of next state vectors
                # 4i-3 and 4i-2
                norm_coeff_p1=[(1-action)*model_s_t,math.sqrt(0.01*(1-action)+0.25*action)]
                #coeffs for 4i-1 and 4i
                norm_coeff_p2=[action*model_s_t,math.sqrt(0.01*action+0.25*(1-action))]

                #Populate the state variables by sampling from the normal
                #distribution params specified above
                new_state[4*(i)-1]=np.random.normal(loc=norm_coeff_p2[0],scale=norm_coeff_p2[1],size=1)[0]
                new_state[4*(i)-1-1]=np.random.normal(loc=norm_coeff_p2[0],scale=norm_coeff_p2[1],size=1)[0]
                new_state[4*(i)-2-1]=np.random.normal(loc=norm_coeff_p1[0],scale=norm_coeff_p1[1],size=1)[0]
                new_state[4*(i)-3-1]=np.random.normal(loc=norm_coeff_p1[0],scale=norm_coeff_p1[1],size=1)[0]
            return new_state

    def get_action(self):
        return np.random.binomial(1,0.5,1)[0]

#Define experience buffer class
class experience_replay:
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self,size):
        return np.array(random.sample(self.buffer,size))
    def get_size(self):
        return len(self.buffer)

class Qnetwork:
    def __init__(self):
        #These lines establish the feed-forward part of the network used to choose actions
        self.input_states = tf.placeholder(shape=[None,64],dtype=tf.float32,name="input1")

        #Layer 1
        self.W1 = tf.Variable(tf.random_uniform([64,16],0,0.01))
        self.Qout1 = tf.nn.relu(tf.matmul(self.input_states,self.W1))

        #Layer 2
        self.W2 = tf.Variable(tf.random_uniform([16,2],0,0.01))
        self.Qout3 = tf.matmul(self.Qout1,self.W2)

        # #Layer3
        # self.W3 = tf.Variable(tf.random_uniform([16,2],0,0.01))
        # self.Qout3 = tf.matmul(self.Qout2,self.W3)

        self.predict = tf.argmax(self.Qout3,1,name="op_to_restore") # Remember to replace the Qout here with the most recent Qout
        self.nextQ = tf.placeholder(shape=[None,2],dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.square(self.nextQ - self.Qout3))# Remember to replace the Qout here with the most recent Qout

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.updateModel = self.trainer.minimize(self.loss)

#Initialize network parameters
batch_size=30
gamma=0.5
num_episodes=5001
environment_steps=91
e=0.1
min_buffer_size=200
tf.reset_default_graph()
Qnet=Qnetwork()
init = tf.initialize_all_variables()
experience_buffer=experience_replay()
saver=tf.train.Saver(max_to_keep=0)
rList=[]
with tf.Session() as sess:
    sess.run(init)
    for it in range(num_episodes):
        s=FOMS()
        s1=s.get_next_state([],1)
        next_state=s1
        rList.append(0)
        if it%100==0 and it!=0:
            save_path=saver.save(sess,'models/my-model',global_step=it)
            #next_state=[]
        for t in range(environment_steps):
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([Qnet.predict,Qnet.Qout3],feed_dict={Qnet.input_states:np.reshape(next_state,(1,64))})#Replace Q-out here with most recent Qout
            if np.random.rand(1) < e:
                a[0] = s.get_action()
            if t==0:
                new_state=s.get_next_state(s1,a[0])
                reward=s.get_reward(s1,a[0])
                experience_tuple=[s1,a[0],reward,new_state]
            else:
                new_state=s.get_next_state(next_state,a[0])
                reward=s.get_reward(next_state,a[0])
                experience_tuple=[next_state,a[0],reward,new_state]
            rList[it]+=reward
            #Add tuple <state,action,reward,new state> to experience buffer
            experience_buffer.add(experience_tuple)

            if experience_buffer.get_size()>min_buffer_size:
                experience_batch=experience_buffer.sample(batch_size)
                #Code to calculate targetQ and then perform gradient descent and update model
                #Read original implementation and work out finer details
                first_state_array=np.reshape(np.vstack(experience_batch[:,0]),(-1,64))
                next_state_array=np.vstack(experience_batch[:,3])
                rewards_array=np.squeeze(np.vstack(experience_batch[:,2]))
                current_Q_arr=sess.run(Qnet.Qout3,feed_dict={Qnet.input_states: np.reshape(np.vstack(experience_batch[:,0]),(-1,64))})
                next_Q_arr=sess.run(Qnet.Qout3,feed_dict={Qnet.input_states: np.reshape(np.vstack(experience_batch[:,3]),(-1,64))})
                y=np.add(np.multiply(np.amax(next_Q_arr,axis=1),gamma),rewards_array)
                y_prop=np.zeros((batch_size,2))
                for i in range(batch_size):
                    current_Q_arr[i][experience_batch[i,1]]=y[i]
                    #y_prop[i,experience_batch[i,1]]=y[i]

                _=sess.run([Qnet.updateModel],feed_dict={Qnet.input_states: first_state_array,Qnet.nextQ:current_Q_arr})
            #Once all is done, set next state equal to new state
            next_state=new_state
