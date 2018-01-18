import numpy as np
import tensorflow as tf
import math
from simulation import Simulation
import json
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

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    new_saver = tf.train.import_meta_graph('./my_dqn.ckpt.meta')
    new_saver.restore(sess,'./my_dqn.ckpt')
    graph=tf.get_default_graph()
    for op in tf.all_variables():
        print op

#     inp=graph.get_tensor_by_name("input1:0")
#     op_to_restore=graph.get_tensor_by_name("op_to_restore:0")
#     gamma=0.6
#     net_glitazone_opp=0
#     reward_sum=0
#     det_glitazone_aug=0
#     net_glitazone_aug=0
#     freq_insulin_aug=0
#     freq_insulin_aug_disc=0
#     #rewards_dict={"0.0":0,"1.0":0,"-2.0":0,"-10.0":0}
#     for strt_pt in gold_standard_dict:
#         rList=[]
#         starting_NAT=int(strt_pt.split(",")[0])
#         starting_D=int(strt_pt.split(",")[1])
#         starting_A1c_cat=int(strt_pt.split(",")[2])
#         for i in range(500):
#             sim=Simulation()
#             first_state, death_indicator=sim.get_custom_first_state(starting_NAT,starting_D,starting_A1c_cat)
# 			#Dynamic variable to keep track of current state
#             current_state=first_state
#             rList.append(0)
#             for t in range(20):
#                 #Abandon trajactory if patient dies
#                 if death_indicator==1:
#                     break
#                 #Take deterministic action if next action is deterministic
#                 elif sim.is_next_action_deterministic(current_state):
#                     action=sim.get_next_deterministic_action(current_state)
#                     if current_state[0]==0 and action==1:
# 						det_glitazone_aug+=1
# 		    #print "Deterministic Action:" +str(action)
#                     next_state,death_indicator=sim.get_next_state(current_state,action)
#                     reward=sim.get_reward(next_state,death_indicator,0)
# 		    #rewards_dict[str(reward)]+=1
#                     rList[i]+=(gamma**t)*reward
#                     #Update state tracking variable reflect current state
#                     current_state=next_state
#                 else:
#                     #Choose an action by greedily (with e chance of random action) from the Q-network
#                     a= sess.run([op_to_restore],feed_dict={inp:np.reshape(current_state,(1,5))})#Replace Q-out here with most recent Qout
# 		    if current_state[0]==2:
# 		    	net_glitazone_opp+=1
# 		    if current_state[0]==2 and a[0][0]==1:
# 				net_glitazone_aug+=1
# 		    #print "Network Chosen Action: "+str(a[0])
#                     #print "Action: "+str(a[0][0])
#                     next_state,death_indicator=sim.get_next_state(current_state,a[0][0])
#                     reward=float(sim.get_reward(next_state,death_indicator,0))
# 		    #rewards_dict[str(reward)]+=1
# 		    #print "Reward: "+str(reward)
#                     rList[i]+=(gamma**t)*reward
#                     #Once all is done, set next state equal to new state
#                     current_state=next_state
#         print "Starting Point:"+strt_pt+" "+str(np.mean(rList))
#         gold_standard_dict[strt_pt][2]=np.mean(rList)
# #print rewards_dict
# print "Deterministic Glitazone Augment: "+str(det_glitazone_aug)
# print "Network Glitazone Augment: "+str(net_glitazone_aug)
# results_json=json.dumps(gold_standard_dict)
# with open("results_json_corrected.txt", 'w+') as outfile:
# 	outfile.write(results_json)
# print float(net_glitazone_aug)/float(net_glitazone_opp)
