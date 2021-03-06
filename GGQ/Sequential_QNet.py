import numpy as np
import math
import tensorflow as tf
import random
from simulation import Simulation
from erbuffer import experience_replay
from Qnetwork import Qnetwork
import json

def measure_NAT_breakdown(training_batch):
	r_0=0
	r_1=0
	r_neg_2=0
	r_neg_10=0
	for row in training_batch:
		if row[3]==0:
			r_0+=1
		if row[3]==-2:
			r_neg_2+=1
		if row[3]==1:
			r_1+=1
		if row[3]==-10:
			r_neg_10+=1
	return r_0,r_1,r_neg_2,r_neg_10

#Create dictionary containing gold standard < starting point: ValGGQ, ValBell, NN_output>
gold_standard_dict={"0,0,2":[1.71186300,1.63361898,0],
					"1,0,2":[1.83348779,1.71002088,0],
					"2,0,2":[1.29620798,1.23576843,0],
					"3,0,2":[1.43314254,1.28809221,0],
					"1,1,2":[1.64399895,1.38670350,0],
					"2,1,2":[0.85822660,0.71743400,0],
					"3,1,2":[1.47778867,1.24709925,0],
					"0,0,3":[1.58978070,1.52902093,0],
					"1,0,3":[1.66901804,1.64977201,0],
					"2,0,3":[0.99612731,0.96004300,0],
					"3,0,3":[1.26827553,1.09084693,0],
					"1,1,3":[1.52392711,1.10862589,0],
					"2,1,3":[0.58233186,0.39627025,0],
					"3,1,3":[1.23757631,0.83849371,0],
					"0,0,4":[1.42174333,1.32456783,0],
					"1,0,4":[1.55630503,1.16993760,0],
					"2,0,4":[0.62467619,0.45342046,0],
					"3,0,4":[0.80732423,0.61960118,0],
					"1,1,4":[1.39808209,0.56663832,0],
					"2,1,4":[0.18776533,0.05791834,0],
					"3,1,4":[0.78017752,0.50054326,0],
					"0,0,5":[1.33540315,1.16855624,0],
					"1,0,5":[1.37212009,1.32693681,0],
					"2,0,5":[0.29191400,0.26016290,0],
					"3,0,5":[0.48464258,0.28612816,0],
					"1,1,5":[1.23999688,0.64637385,0],
					"2,1,5":[-0.01053616,-0.21975549,0],
					"3,1,5":[0.47821270,0.11402890,0]}
#Initialize network parameters
batch_size=64
gamma=0.6
num_episodes=5001
environment_steps=20
e=0.1
min_buffer_size=100

#Initialize tensoflow graph
tf.reset_default_graph()
Qnet=Qnetwork()
init = tf.initialize_all_variables()

#Initialize experience buffer
experience_buffer=experience_replay(min_buffer_size)

#Model Saver
saver=tf.train.Saver(max_to_keep=0)

#Summary Saver

#Reward List
rList=[]
step=0

# Record training examples distribution
NAT_dict={"0":0,"1":0,"-2":0,"-10":0}
with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter("./logs/",sess.graph)
	#Evaluate network every 50 steps
	max_reward=-100
	for it in range(num_episodes):
		#Initialize simulation
		sim=Simulation()
		first_state,death_indicator=sim.get_first_state()

		#Dynamic variable to keep track of current state
		current_state=first_state
		current_death_indicator=death_indicator
		#Model Checkpoint
		if it%100==0 and it!=0:
			#save_path=saver.save(sess,'models/my-model',global_step=it)
			continue
		#Evaluate network every 50 steps
		if step%100==0 and step!=0:
			epoch_reward=0
			for strt_pt in gold_standard_dict:
				starting_NAT=int(strt_pt.split(",")[0])
				starting_D=int(strt_pt.split(",")[1])
				starting_A1c_cat=int(strt_pt.split(",")[2])
				#print "("+str(starting_NAT)+","+str(starting_D)+","+str(starting_A1c_cat)+")"
				gamma=0.6
				eval_rList=[]
				for index in range(500):
					eval_sim=Simulation()
					eval_first_state,eval_death_indicator=eval_sim.get_custom_first_state(starting_NAT,starting_D,starting_A1c_cat)
					#print eval_first_state
					#Dynamic variable to keep track of current state
					eval_current_state=eval_first_state
					eval_rList.append(0)
					for time in range(20):
						#print eval_current_state
						#Abandon trajactory if patient dies
						if eval_death_indicator==1:
							break
							#Take deterministic action if next action is deterministic
						elif eval_sim.is_next_action_deterministic(eval_current_state):
							eval_action=eval_sim.get_next_deterministic_action(eval_current_state)
							#print "Deterministic Action:" +str(action)
							eval_next_state,eval_death_indicator=eval_sim.get_next_state(eval_current_state,eval_action)
							eval_reward=float(eval_sim.get_reward(eval_next_state,eval_death_indicator))
							#print "Reward: "+str(eval_reward)
							eval_rList[index]+=(gamma**time)*eval_reward
							#Update state tracking variable reflect current state
							eval_current_state=eval_next_state
						else:
							#Choose an action by greedily (with e chance of random action) from the Q-network
							eval_a= sess.run([Qnet.predict],feed_dict={Qnet.input_states:np.reshape(eval_current_state,(1,5))})#Replace Q-out here with most recent Qout
							#print "Network Chosen Action: "+str(a[0])
							#print "Action: "+str(a[0][0])
							eval_next_state,eval_death_indicator=eval_sim.get_next_state(eval_current_state,eval_a[0][0])
							eval_reward=float(eval_sim.get_reward(eval_next_state,eval_death_indicator))
							#print "Reward: "+str(eval_reward)
							eval_rList[index]+=(gamma**time)*eval_reward
							#Once all is done, set next state equal to new state
							eval_current_state=eval_next_state
				point_reward=np.mean(eval_rList)
				#print "("+str(starting_NAT)+","+str(starting_D)+","+str(starting_A1c_cat)+"): "+str(point_reward)
				epoch_reward+=point_reward
			if epoch_reward>max_reward:
				max_reward=epoch_reward
				save_path=saver.save(sess,'models/best_model')
				print step
				print epoch_reward

		# SDG and optimization
		for t in range(environment_steps):
			#Abandon trajactory if patient dies
			if current_death_indicator==1:
				break
			# Check if action is deterministic, excluding it from the buffer
			elif sim.is_next_action_deterministic(current_state):
				a=sim.get_next_deterministic_action(current_state)
				next_state,next_death_indicator=sim.get_next_state(current_state,a)
				reward=sim.get_reward(next_state,next_death_indicator)
				#Code to calculate targetQ and then perform gradient descent and update mo
                        	#Q-values for succeeding states according to the metwork as it is now
                        	next_Q_arr=sess.run(Qnet.Qout3,feed_dict={Qnet.input_states:np.reshape(next_state,(1,5))})
                        	#Get the value of y= r+gamma*Q
                        	y=np.add(np.multiply(np.amax(next_Q_arr[0]),gamma),reward)
                        	#Q-values according to the network as it is now
                        	updated_Q_arr=sess.run(Qnet.Qout3,feed_dict={Qnet.input_states: np.reshape(current_state,(1,5))})
                        	#Update Q array with r+gamma*Q  and run least squares minmization against first_state_array
                        	updated_Q_arr[0][int(a)]=y
                        	_,summary,loss=sess.run([Qnet.updateModel,Qnet.summary_op,Qnet.loss],feed_dict={Qnet.input_states:np.reshape(current_state,(1,5)),Qnet.nextQ:updated_Q_arr})
                        	writer.add_summary(summary,step)
                        	step+=1
			else:
				#Choose an action by greedily (with e chance of random action) from the Q-network
				a,allQ = sess.run([Qnet.predict,Qnet.Qout3],feed_dict={Qnet.input_states:np.reshape(current_state,(1,5))})
				if random.random() < e:
					a[0] = np.random.randint(2,size=1)[0]
				next_state,next_death_indicator=sim.get_next_state(current_state,a[0])
				reward=sim.get_reward(next_state,next_death_indicator)
				#Code to calculate targetQ and then perform gradient descent and update mo
                        	#Q-values for succeeding states according to the metwork as it is now
                        	next_Q_arr=sess.run(Qnet.Qout3,feed_dict={Qnet.input_states:np.reshape(next_state,(1,5))})
                       		#Get the value of y= r+gamma*Q
                        	y=np.add(np.multiply(np.amax(next_Q_arr[0]),gamma),reward)
                        	#Q-values according to the network as it is now
                        	updated_Q_arr=sess.run(Qnet.Qout3,feed_dict={Qnet.input_states: np.reshape(current_state,(1,5))})
                        	#Update Q array with r+gamma*Q  and run least squares minmization against first_state_array
                        	updated_Q_arr[0][a[0]]=y
                        	_,summary,loss=sess.run([Qnet.updateModel,Qnet.summary_op,Qnet.loss],feed_dict={Qnet.input_states:np.reshape(current_state,(1,5)),Qnet.nextQ:updated_Q_arr})
                        	writer.add_summary(summary,step)
                        	step+=1

			
			#Once all is done, set next state equal to new state
                	current_state=next_state
                	current_death_indicator=next_death_indicator
