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

def eval_net(sess,gold_standard_dict,Qnet,saver,max_reward,step,Q_dict):
	epoch_reward=0
	network_augments=0
	network_opp=0
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
			#Dynamic variable to keep track of current state
			eval_current_state=eval_first_state
			eval_rList.append(0)
			for time in range(10):
				#Abandon trajactory if patient dies
				if eval_death_indicator==1:
					break
					#Take deterministic action if next action is deterministic
				elif eval_sim.is_next_action_deterministic(eval_current_state):
					eval_action=eval_sim.get_next_deterministic_action(eval_current_state)
					eval_next_state,eval_death_indicator=eval_sim.get_next_state(eval_current_state,eval_action)
					eval_reward=float(eval_sim.get_reward(eval_next_state,eval_death_indicator,0))
					eval_rList[index]+=(gamma**time)*eval_reward
					#Update state tracking variable reflect current state
					eval_current_state=eval_next_state
				else:
					#Choose an action by greedily (with e chance of random action) from the Q-network
					network_opp+=1
					eval_a= sess.run([Qnet.predict],feed_dict={Qnet.input_states:np.reshape(eval_current_state,(1,5))})
					if eval_a[0][0]==1:
						network_augments+=1
					eval_next_state,eval_death_indicator=eval_sim.get_next_state(eval_current_state,eval_a[0][0])
					eval_reward=float(eval_sim.get_reward(eval_next_state,eval_death_indicator,0))
					eval_rList[index]+=(gamma**time)*eval_reward
					#Once all is done, set next state equal to new state
					eval_current_state=eval_next_state
		point_reward=np.mean(eval_rList)
		Q_dict[strt_pt].append(float(point_reward))
		epoch_reward+=point_reward
	print "Epoch Reward:"+str(epoch_reward)
	#print str(float(network_augments)/float(network_opp))
	if epoch_reward>max_reward:
		max_reward=epoch_reward
		save_path=saver.save(sess,'models/best_model')
		#print step
		#print epoch_reward
	return max_reward

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
# Dictionary to record how Q-value progresses over time for individual states
state_Q_dict={"0,0,2":[],
 			  "1,0,2":[],
              "2,0,2":[],
 		  	  "3,0,2":[],
 		  	  "1,1,2":[],
 		  	  "2,1,2":[],
 		      "3,1,2":[],
 		  	  "0,0,3":[],
 		      "1,0,3":[],
 		  	  "2,0,3":[],
 		  	  "3,0,3":[],
 		  	  "1,1,3":[],
 		  	  "2,1,3":[],
   			  "3,1,3":[],
 		      "0,0,4":[],
 		  	  "1,0,4":[],
 		  	  "2,0,4":[],
  	  		  "3,0,4":[],
   			  "1,1,4":[],
 		  	  "2,1,4":[],
 		  	  "3,1,4":[],
 		  	  "0,0,5":[],
 		  	  "1,0,5":[],
 		  	  "2,0,5":[],
 		  	  "3,0,5":[],
 		  	  "1,1,5":[],
 		  	  "2,1,5":[],
 		  	  "3,1,5":[]}
#Initialize network parameters
batch_size=64
gamma=0.6
num_episodes=10001
environment_steps=20
e=0.1
min_buffer_size=20000

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

#Loss list tracker
loss_list=[]

#
# Record training examples distribution
with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter("./logs/",sess.graph)
	#saver.restore(sess,'./models/best_model')
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
			save_path=saver.save(sess,'models/my-model',global_step=it)
			continue
		# SDG and optimization
		for t in range(environment_steps):
			#Abandon trajactory if patient dies
			if current_death_indicator==1:
				break
			# Check if action is deterministic, excluding it from the buffer
			elif sim.is_next_action_deterministic(current_state):
				a=sim.get_next_deterministic_action(current_state)
				next_state,next_death_indicator=sim.get_next_state(current_state,a)
				reward=sim.get_reward(next_state,next_death_indicator,a)
				experience_tuple=[current_state,current_death_indicator,a,reward,next_state,next_death_indicator]
				experience_buffer.add(experience_tuple)
			else:
				#Choose an action by greedily (with e chance of random action) from the Q-network
				a,allQ = sess.run([Qnet.predict,Qnet.Qout3],feed_dict={Qnet.input_states:np.reshape(current_state,(1,5))})
				if random.random() < e:
					a[0] = np.random.randint(2,size=1)[0]
				next_state,next_death_indicator=sim.get_next_state(current_state,a[0])
				reward=sim.get_reward(next_state,next_death_indicator,a[0])
				experience_tuple=[current_state,current_death_indicator,a[0],reward,next_state,next_death_indicator]
				#Add tuple <state,action,reward,new state> to experience buffer
				experience_buffer.add(experience_tuple)
			#Once all is done, set next state equal to new state
	            	current_state=next_state
            		current_death_indicator=next_death_indicator
		if experience_buffer.get_size()>min_buffer_size:
			experience_batch=experience_buffer.sample(batch_size)
			#Code to calculate targetQ and then perform gradient descent and update model
			#Initial state
			state_array=np.reshape(np.vstack(experience_batch[:,0]),(-1,5))
			#States after taking action
			succeeding_state_array=np.vstack(experience_batch[:,4])
			#Terminality of suceeding state
			succeeding_state_terminality=np.vstack(experience_batch[:,-1])
			#Array storing rewards
			rewards_array=np.squeeze(np.vstack(experience_batch[:,3]))
			#Array storing actions
			actions_array=np.squeeze(np.vstack(experience_batch[:,2]))
			# This is where difference between deterministic state and network states needs to be accounted for
			# y is a list of the form [update Q value, action taken]
			y=[]
			#Compute update data
			for i in range(batch_size):
				y.append([0,0])
				if succeeding_state_terminality[i]==1:
					y[i]=rewards_array[i]
				else:
					#Q-values for succeeding states according to the network as it is now
					next_Q_arr=sess.run(Qnet.Qout3,feed_dict={Qnet.input_states: np.reshape(succeeding_state_array[i],(-1,5))})
					if sim.is_next_action_deterministic(succeeding_state_array[i]):
						action=sim.get_next_deterministic_action(succeeding_state_array[i])
					else:
						action=np.argmax(next_Q_arr[0])
					#Get the value of y= r+gamma*Q
					y[i]=np.add(np.multiply(next_Q_arr[0,action],gamma),rewards_array[i])
			#Q-values according to the network as it is now
			#online_q_values=sess.run(Qnet.Qout3,feed_dict={Qnet.input_states: state_array})
			_,summary,loss,online_q_values=sess.run([Qnet.updateModel,Qnet.summary_op,Qnet.loss,Qnet.online_q_values],feed_dict={Qnet.input_states:state_array,Qnet.actions_array:actions_array,Qnet.y:y})
			print loss
			loss_list.append(float(loss))
			writer.add_summary(summary,step)
			step+=1
			#Evaluate network every 50 steps
			if step%50==0 and step!=0:
				max_reward=eval_net(sess,gold_standard_dict,Qnet,saver,max_reward,step,state_Q_dict)
# with open("./reward_breakdown.txt","wb+") as reward_file:
# 	rewards_json=json.dumps(NAT_dict)
# 	reward_file.write(rewards_json)
# with open("./loss_tracker.txt","wb+") as loss_file:
# 	loss_json=json.dumps(loss_list)
# 	loss_file.write(loss_json)
#
# with open("./Q_tracker.txt","wb+") as Q_file:
# 	Q_json=json.dumps(state_Q_dict)
# 	Q_file.write(Q_json)
# print A1c_dict
