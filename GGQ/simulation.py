import numpy as np
import math
class Simulation():
	def __init__(self):
		self.metrformin_te=0.14
		self.solfonylurea_te=0.2
		self.glitazone_te=0.02
		self.insulin_te=0.14
		self.sd_eps=0.5
		self.prev_u_t=9

	#return tuple of < NAT, D, A1c, BP, Weight>,C_t at beginning of trajectory
	def get_first_state(self):
		bp_0=np.random.normal(0,1,1)[0]
		w_0=np.random.normal(0,1,1)[0]
		A1c_0=np.random.normal(9,1,1)[0]
		s_0=[0,0,A1c_0,bp_0,w_0]
		return s_0,0

	def get_next_state(self,prev_state,action):
		eps=np.random.normal(0,self.sd_eps,1)
		bp_t=self.get_bp_t(prev_state[3],eps)[0]
		w_t=self.get_w_t(prev_state[4],eps)[0]
		c_t=self.get_c_t(prev_state[2],prev_state[0])[0]
		#Initialize NAT
		NAT=0
		if prev_state[0]<4:
			if action==1:
				NAT=prev_state[0]+1
			else:
				NAT=prev_state[0]
			d_t=self.get_d_t(action,prev_state[0])[0]
			A1c_t=self.get_A1c(prev_state,action,d_t,eps)[0]
			return [NAT,d_t,A1c_t,bp_t,w_t],c_t
		# If NAT>4, make algorithm indiffernt between action 1 and 0 to enable better state space exploration
		else:
			d_t=self.get_d_t(0,prev_state[0])[0]
			A1c_t=self.get_A1c(prev_state,0,d_t,eps)[0]
			return [4,d_t,A1c_t,bp_t,w_t],c_t

	#Return tuple based on custom soecifications of NAT,D,cat.A1c
  	def get_custom_first_state(self,starting_NAT, starting_D,starting_A1c_cat):
    		bp=np.random.normal(0,1,1)[0]
    		w=np.random.normal(0,1,1)[0]
    		if starting_A1c_cat==2:
      			A1c=np.random.uniform(7,7.2,1)[0]
    		elif starting_A1c_cat==3:
      			A1c=np.random.uniform(7.2,7.5,1)[0]
    		elif starting_A1c_cat==4:
      			A1c=np.random.uniform(7.5,7.7,1)[0]
    		elif starting_A1c_cat==5:
      			A1c=np.random.uniform(7.7,8,1)[0]
    		return [starting_NAT,starting_D,A1c,bp,w],0

	def get_bp_t(self,prev_bp,eps):
		bp_t=(prev_bp+eps)/math.sqrt(1+float(self.sd_eps)**2)
		return bp_t

	def get_w_t(self,prev_w,eps):
		w_t=(prev_w+eps)/math.sqrt(1+float(self.sd_eps)**2)
		return w_t
	def get_c_t(self,prev_A1c,prev_NAT):
		A1c_indicator=int(prev_A1c>7)
		x=-10+0.08*A1c_indicator*(prev_A1c**2)+0.5*prev_NAT
		c_t=np.random.binomial(1,self.exp_helper(x),1)
		return c_t

	def get_d_t(self,action,prev_NAT):
		p=0
		if action==0:
			return [0]
		else:
			if prev_NAT==3:
				p=0.35
			else:
				p=0.2
			d_t=np.random.binomial(1,p,1)
			return d_t

	def get_A1c(self,prev_state,action,d_t,eps):
		if prev_state[2]>7 and prev_state[0]<4 and action!=0 and d_t!=1:
			new_u_t=0
			if prev_state[0]==0:
				new_u_t=self.prev_u_t*(1-self.metrformin_te)
			elif prev_state[0]==1:
				new_u_t=self.prev_u_t*(1-self.solfonylurea_te)
			elif prev_state[0]==2:
				new_u_t=self.prev_u_t*(1-self.glitazone_te)
			elif prev_state[0]==3:
				new_u_t=self.prev_u_t*(1-self.insulin_te)
			A1c_t=(prev_state[2]-self.prev_u_t+eps)/math.sqrt(1+self.sd_eps**2)+new_u_t
			self.prev_u_t=new_u_t
			return A1c_t
		else:
			A1c_t=(prev_state[2]-self.prev_u_t+eps)/math.sqrt(1+self.sd_eps**2)+self.prev_u_t
			return A1c_t

	def get_reward(self,new_state,C_t,action):
		if C_t==1:
			return -10.0
		elif new_state[2]<7 and action==0:
			return 1.0
		elif new_state[2]<7 and action==1:
			return 5.0
		elif new_state[2]>7 and new_state[1]==1:
			return -2.0
		else:
			return 0.0

	def is_next_action_deterministic(self,state):
		if state[0]<4:
			if state[2]<7:
				return True
			elif state[2]>8:
				return True
			else:
				return False
		# If NAT>4, make algorithm indiffernt between action 1 and 0 to enable better state space exploration
		else:
			return True
	def get_next_deterministic_action(self,state):
		if state[0]<4:
			if state[2]<7:
				return 0
			elif state[2]>8:
				return 1
		# If NAT>4, make algorithm indiffernt between action 1 and 0 to enable better state space exploration
		else:
			return 0

	#Helper function to calculate value of e^x/(1+e^x)
	def exp_helper(self,x):
		return float(math.exp(x))/float(1+math.exp(x))
