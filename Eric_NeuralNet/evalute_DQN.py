import tensorflow as tf
import numpy as np
import math

class FOMS:

    def get_reward(self,state,action):
        normal_coeff_p1=2*(state[0]+state[1])-(state[2]+state[3])
        normal_coeff_p2=2*(state[2]+state[3])-(state[0]+state[1])
        return np.random.normal(loc=(1-action)*(normal_coeff_p1)+action*(normal_coeff_p2),scale=0.01,size=1)[0]

    def get_next_state(self,current_state,action):
        if len(current_state)==0:
            return np.random.normal(loc=0,scale=0.25,size=64)
        else:
            new_state=np.zeros(64)
            for i in range(1,17):
                s_t=current_state[i-1]
                model_s_t=s_t

                #mean and sd for normal distribution of next state vectors
                # 4i-3 and 4i-2
                norm_coeff_p1=[(1-action)*model_s_t,0.01*(1-action)+0.25*action]
                #coeffs for 4i-1 and 4i
                norm_coeff_p2=[action*model_s_t,0.01*action+0.25*(1-action)]

                #Populate the state variables by sampling from the normal
                #distribution params specified above
                new_state[4*(i)-1]=np.random.normal(loc=norm_coeff_p2[0],scale=norm_coeff_p2[1],size=1)[0]
                new_state[4*(i)-1-1]=np.random.normal(loc=norm_coeff_p2[0],scale=norm_coeff_p2[1],size=1)[0]
                new_state[4*(i)-2-1]=np.random.normal(loc=norm_coeff_p1[0],scale=norm_coeff_p1[1],size=1)[0]
                new_state[4*(i)-3-1]=np.random.normal(loc=norm_coeff_p1[0],scale=norm_coeff_p1[1],size=1)[0]
            return new_state

    def get_action(self):
        return np.random.binomial(1,0.5,1)[0]

init = tf.initialize_all_variables()
with tf.Session() as sess:
  sess.run(init)
  new_saver = tf.train.import_meta_graph('./models/my-model-500.meta')
  new_saver.restore(sess,'./models/my-model-500')
  graph=tf.get_default_graph()
  inp=graph.get_tensor_by_name("input1:0")
  op_to_restore=graph.get_tensor_by_name("op_to_restore:0")
  rList=[]
  gamma=0.5
  total_decisions=0
  correct_actions=0
  for i in range(30):
    sim=FOMS()
    s1=sim.get_next_state([],1)
    rList.append(0)
    for j in range(90):
        reward_action_0=sim.get_reward(s1,0)
        reward_action_1=sim.get_reward(s1,1)
        action=0
        if reward_action_1>reward_action_0:
            action=1
            reward=reward_action_1
        else:
            reward=reward_action_0
        #action=sess.run([op_to_restore],feed_dict={inp:np.reshape(s1,(1,64))})
        # if action[0][0]==action:
        #     correct_actions+=1
        #reward=sim.get_reward(s1,action[0][0])
        rList[i]+=reward*(gamma**j)
        s1=sim.get_next_state(s1,action)
        total_decisions+=1
print "Mean reward over 30 iterations:"+str(np.mean(rList))
print "Percentage of correct decisions:"+ str(correct_actions*100/float(total_decisions))
