import numpy as np
import random
#Define experience buffer class
class experience_replay:
	def __init__(self, buffer_size = 1000):
		self.buffer = []
		self.buffer_size = buffer_size
	def add(self,experience):
		#Keep buffer_size number of most recent operations
		if len(self.buffer)>self.buffer_size:
			self.buffer= self.buffer[-1*(self.buffer_size-1):]
		self.buffer.append(experience)
	def sample(self,size):
		return np.array(random.sample(self.buffer,size))
	def get_size(self):
		return len(self.buffer)
