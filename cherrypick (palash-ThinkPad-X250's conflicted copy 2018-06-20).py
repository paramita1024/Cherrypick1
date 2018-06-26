import numpy.ma as ma
import numpy as np
from slant.py import slant
class performance:
	def __init__(self):
		self.MSE=0
		self.FR=0	

class slant_input_data:
	def __init__(self, nodes, edges, test, train):

		# check slant input
		# create it accordingly
		# pass to slant 
		self.nodes = nodes
		self.edges = edges
		self.test = test
		self.train = train

		# init with obj 
		# edges
		# train
		# test
		# nodes
		self.nodes = obj.nodes
		self.edges = obj.edges
		self.test = obj.test
		self.train = obj.train

	# 	# split each user list in test and train

	# 	with open(filename,'rb') as f:
	# 		data = pickle.load(f)
	# 		# data is a class containing graph, test and train
	# 	self.graph = data.graph
	# 	self.train = data.train
	# 	self.test = data.test
	def find_argmax(self, nodes_end, nodes_exo, msg_end, msg_exo):
		users_of_end_msgs = self.train[ msg_end, 0 ]
		max_inc = - Inf
		for user in nodes_exo.nonzero()[0]:
			index_user = np.where(users_of_end_msgs == user)[0]
			if index_user.shape[0] > 0 :
				msg_end_indices #
					flag_change_user = True
			for msg in msg_exo.nonzero()[0]:
	 		user_msg_end = msg_end_indices[ index_user ]
				# add msg of those indices
				flag_change_user = True
			for msg in msg_exo.nonzero()[0]:
				user_curr, time_curr, sentiment_curr = self.train[msg,:]
				if nodes_end[user_curr] | user_curr == user:
					# add info of this msg
					flag_change_msg = True
				if flag_change_msg | flag_change_user:
					# compute change or current increment
					# if it is current max, set that
					if current_inc > max_inc:
						max_user = user
						max_msg_no = msg
						max_inc = current_inc

				flag_change_msg  = False
			flag_change_user = False
		return max_msg_no , max_user
	def demarkate_process(self, frac_nodes_end, frac_msg_end):
		#-------------------------------------------------
		# create two mat, msg_end, msg_exo
		# nodes_end , nodes_exo
		# frac_nodes_end , frac_msg_end
		# return nodes_end , msg_end
		#-------------------------------------------------
		max_end_user = int(frac_nodes_end*self.num_nodes)
		max_end_msg = int(frac_msg_end * self.num_train)

		nodes_end = ma.make_mask(np.zeros(self.num_nodes)) 
		nodes_exo = ma.make_mask(np.ones(self.num_nodes))
		msg_end = ma.make_mask(np.zeros(self.num_train)) 
		msg_exo = ma.make_mask(np.ones(self.num_train)) 

		while  msg_end.count() < max_end_msg: # check count
			if nodes_end.count() < max_end_user:
				msg_no , user = self.find_argmax(nodes_end, nodes_exo, msg_end, msg_exo)
				nodes_exo[user] = False #.remove(user)
				nodes_end[user] = True #.insert(user)
			else:
				msg_no = self.find_argmax()
			msg_exo[ msg_no ] = False #.remove(msg)
			msg_end[ msg_no ] = True #.insert(msg)

		self.mask_nodes_end  = nodes_end
		self.mask_msg_end= msg_end
		return  











		# # init H,V,O,I
		# H=[]
		# V=range(ntrain)
		# O=[]
		# I=range(nuser)
		# # number of user not exceeded, 
		# while len(O) <= self.max_end_user:
		# 	# select msg and user 
		# 	H.append(m)
		# 	V.remove(m)
		# 	O.append(u)
		# 	I.remove(u)

		
		# # while msg limit has not reached
		# # select a msg 
		# # include in H , exclude from V
		# # return H,O
		# while len(H) <= self.max_end_msg:
		# 	# select m
		# 	H.append(m)
		# 	V.remove(m)
		# return H,V,O,I
	def evaluate_using_slant(self):
		#----------------------------------------------------------
		# init slant obj
		# init slant 
		# call slant estimate
		# call evaluate method of slant
		#----------------------------------------------------------
		slant_input_data_obj = slant_input_data( self.nodes, self.edges, self.msg_end , self.test, flag_evaluate_real = True)
		slant_obj = slant( slant_input_data_obj )
		slant_obj.estimate()
		result  =  slant_obj.predict()
		# result : dictionary with two field. field1 = 'MSE', field2 = 'FR'
		return result
	# def train_model(self):
	# 	H,V,O,I = self.find_H_and_O()
	# 	# modify input train test graph
	# 	self.train,self.test, self.train_ex, self.test_ex = self.reduce()
	# 	self.slant_opt= slant(self.graph)
	# 	self.slant_opt.estimate_param(self.train)# define and pass parameters	
	# def forecast(self):
	# 	self.result = self.slant_opt.predict_sentiment(self.test)
	# def create_graph(self):
	# 	self.graph={}
	# 	for v in num_v:
	# 		self.graph[v]=set([])
	# 	for node1,node2 in set_of_egdes:
	# 		self.graph[node1].add(node2)
	# def load(self):
	# 	data=np.genfromtxt(self.fp,delimiter=',')
	# 	user,index,count = np.unique(data,return_index=True, return_counts=True)
	# 	for i in range(nuser):
	# 		tr_idx = np.concatenate([tr_idx, index[i,:np.floor(self.split_ratio*count[i])]])
	# 		te_idx = np.concatenate([tr_idx, index[i,np.floor(self.split_ratio*count[i]):]])
	# 	train=data[tr_idx,:]
	# 	test=data[te_idx,:]
	# 	self.ntrain=train.shape[0]
	# 	self.ntest=test.shape[0]
	# 	self.nuser=user.shape[0]

def main():
	# load
	filename = 'a'
	obj = load(filename)

	# init cherrypick
	cherrypick_obj = cherrypick(obj)
	# call cherrypick_method()
	cherrypick_obj.cherrypick_method() #** param
	# init s;lant obj
	# call slant 
	# call slant estimate
	# call evaluate method of slant
	result = cherrypick_obj.evaluate_using_slant()
	# return the number

if __name__== "__main__":
  main()




#--------------------------------------------------------------------------------------------------------------------------------------------

# class cherrypick:
# 	def __init__(self,filename):
# 		# read the data as a list of (user,msg,time)
# 		# split to create per user sorted list of msg
# 		# split each user list in test and train

# 		with open(filename,'rb') as f:
# 			data = pickle.load(f)
# 			# data is a class containing graph, test and train
# 		self.graph = data.graph
# 		self.train = data.train
# 		self.test = data.test
# 	def find_H_and_O(self):
# 		# init H,V,O,I
# 		H=[]
# 		V=range(ntrain)
# 		O=[]
# 		I=range(nuser)
# 		# number of user not exceeded, 
# 		while len(O) <= self.max_end_user:
# 			# select msg and user 
# 			H.append(m)
# 			V.remove(m)
# 			O.append(u)
# 			I.remove(u)

		
# 		# while msg limit has not reached
# 		# select a msg 
# 		# include in H , exclude from V
# 		# return H,O
# 		while len(H) <= self.max_end_msg:
# 			# select m
# 			H.append(m)
# 			V.remove(m)
# 		return H,V,O,I
# 	def train_model(self):
# 		H,V,O,I = self.find_H_and_O()
# 		# modify input train test graph
# 		self.train,self.test, self.train_ex, self.test_ex = self.reduce()
# 		self.slant_opt= slant(self.graph)
# 		self.slant_opt.estimate_param(self.train)# define and pass parameters	
# 	def forecast(self):
# 		self.result = self.slant_opt.predict_sentiment(self.test)
# 	# def create_graph(self):
# 	# 	self.graph={}
# 	# 	for v in num_v:
# 	# 		self.graph[v]=set([])
# 	# 	for node1,node2 in set_of_egdes:
# 	# 		self.graph[node1].add(node2)
# 	# def load(self):
# 	# 	data=np.genfromtxt(self.fp,delimiter=',')
# 	# 	user,index,count = np.unique(data,return_index=True, return_counts=True)
# 	# 	for i in range(nuser):
# 	# 		tr_idx = np.concatenate([tr_idx, index[i,:np.floor(self.split_ratio*count[i])]])
# 	# 		te_idx = np.concatenate([tr_idx, index[i,np.floor(self.split_ratio*count[i]):]])
# 	# 	train=data[tr_idx,:]
# 	# 	test=data[te_idx,:]
# 	# 	self.ntrain=train.shape[0]
# 	# 	self.ntest=test.shape[0]
# 	# 	self.nuser=user.shape[0]