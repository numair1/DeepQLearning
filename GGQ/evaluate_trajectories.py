import numpy as np
import tensorflow as tf
import math
from simulation import Simulation
import json

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    new_saver = tf.train.import_meta_graph('./models/best_model.meta')
    new_saver.restore(sess,'./models/best_model')
    graph=tf.get_default_graph()
    inp=graph.get_tensor_by_name("input1:0")
    op_to_restore=graph.get_tensor_by_name("op_to_restore:0")
    gamma=0.6
    reward_sum=0
    rList=[]
    for i in range(1):
        sim=Simulation()
        first_state, death_indicator=sim.get_custom_first_state(3,1,5)
        #Dynamic variable to keep track of current state
        current_state=first_state
        rList.append(0)
        for t in range(20):
            print current_state
            #Abandon trajactory if patient dies
            if death_indicator==1:
                break
            #Take deterministic action if next action is deterministic
            elif sim.is_next_action_deterministic(current_state):
                action=sim.get_next_deterministic_action(current_state)
                print "Deterministic Action:" +str(action)
                next_state,death_indicator=sim.get_next_state(current_state,action)
                reward=sim.get_reward(next_state,death_indicator)
                print "Reward: "+str(reward)
                rList[i]+=(gamma**t)*reward
                #Update state tracking variable reflect current state
                current_state=next_state
            else:
                #Choose an action by greedily (with e chance of random action) from the Q-network
                a= sess.run([op_to_restore],feed_dict={inp:np.reshape(current_state,(1,5))})#Replace Q-out here with most recent Qout
                print "Network Chosen Action: "+str(a[0][0])
                #print "Action: "+str(a[0][0])
                next_state,death_indicator=sim.get_next_state(current_state,a[0][0])
                reward=float(sim.get_reward(next_state,death_indicator))
                print "Reward: "+str(reward)
                rList[i]+=(gamma**t)*reward
                #Once all is done, set next state equal to new state
                current_state=next_state
