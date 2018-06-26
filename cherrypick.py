import numpy.ma as ma
import numpy as np
from slant.py import slant
# modify slant to use inheritance

class slant_input_data: # check whether slant will take this input
	def __init__(self, nodes, edges, test, train):

		# check slant input
		# create it accordingly
		# pass to slant 
		self.nodes = nodes
		self.edges = edges
		self.test = test
		self.train = train

		self.num_nodes = self.nodes.shape[0]
		self.num_train = self.train.shape[0]

class cherrypick:
	def __init__(self,obj):
		# init with obj 
		# edges
		# train
		# test
		# nodes
		self.nodes = obj.nodes
		self.edges = obj.edges
		self.test = obj.test
		self.train = obj.train

		self.sigma_covariance = .1

	# 	# read the data as a list of (user,msg,time)
	# 	# split to create per user sorted list of msg
	# 	# split each user list in test and train

	# 	with open(filename,'rb') as f:
	# 		data = pickle.load(f)
	# 		# data is a class containing graph, test and train
	# 	self.graph = data.graph
	# 	self.train = data.train
	# 	self.test = data.test
	# def find_argmax(self, nodes_end, nodes_exo, msg_end, msg_exo):
	# 	users_of_end_msgs = self.train[ msg_end, 0 ]
	# 	max_inc = - Inf
	# 	for user in nodes_exo.nonzero()[0]:
	# 		index_user = np.where(users_of_end_msgs == user)[0]
	# 		if index_user.shape[0] > 0 :
	# 			msg_end_indices #
	# 			user_msg_end = msg_end_indices[ index_user ]
	# 			# add msg of those indices
	# 			flag_change_user = True
	# 		for msg in msg_exo.nonzero()[0]:
	# 			user_curr, time_curr, sentiment_curr = self.train[msg,:]
	# 			if nodes_end[user_curr] | user_curr == user:
	# 				# add info of this msg
	# 				flag_change_msg = True
	# 			if flag_change_msg | flag_change_user:
	# 				# compute change or current increment
	# 				# if it is current max, set that
	# 				if current_inc > max_inc:
	# 					max_user = user
	# 					max_msg_no = msg
	# 					max_inc = current_inc

	# 			flag_change_msg  = False
	# 		flag_change_user = False
	# 	return max_msg_no , max_user
	def create_influence_matrix(self):
		influence_matrix = np.zeros((self.num_nodes+1, self.num_train)) # add 1 # a better idea is to use row major mat instead 
		influence_matrix[0,:] = 1 
		msg_index = 0
		time_old = 0

	
		for user, time, sentiment in self.train : 
			user = int(user)
			if msg_index > 0 :
				influence_matrix[1:,msg_index] = influence_matrix[1:,msg_index-1]*np.exp(-self.w*(time - time_old) ) # use influence_matrix[1:]
			influence_matrix[user, msg_index] += sentiment 
			msg_index += 1
			time_old = time
		return influence_matrix
	# add one here
	def create_covariance_matrix(self):
		self.covariance = np.zeros((self.num_nodes, self.num_nodes+1, self.num_nodes+1))
		for user in self.nodes:
			self.covariance[user, :,:] = c*np.eye( self.num_nodes+1 )
	def evaluate( self, matrix):
		return np.sum(np.log(np.diag(np.inv( matrix ))))# also check the time for this 
	def create_function_val(self):
		self.curr_function_val = np.zeros( self.num_nodes )
		for user in self.nodes:
			self.curr_function_val[user] = self.evaluate( self.covariance[ user ])
	
	def get_influence_matrix_for_msg(self, msg_no):
		influence_vector = self.influence_matrix[:,msg_no]
		msg_mat = (1/self.sigma_covariance^2)*np.matmul(influence_vector, influence_vector.T) # check shape of influence vector or reshape it . save influ as row major. take vector . reshape it. use 
		return msg_mat

	def obtain_most_endogenius_msg_user(self, flag_send_msg_only = False): 
		if flag_send_msg_only:
			inc = np.zeros( self.num_train) 
			inc =  - float('inf')
			for msg_no in self.msg_exo.nonzero()[0]:
				msg_mat = self.get_influence_matrix_for_msg(msg_no)
				user = int(self.train[msg_no,0]) 
				inc[msg_no] =self.evaluate(self.covariance[user] + msg_mat) - self.curr_function_val[user] 
				
			# find max
			msg_to_choose = np.argmax(inc) 
			
			if self.msg_end[msg_to_choose]:
				print "a msg which is already endogenious has been selected again as endogenious msg"

			# return msg no
			return msg_to_choose
		else:
			max_inc = -float('inf')
			user_to_choose = -1
			msg_to_choose = -1
			for user in self.nodes_exo.nonzero()[0]:
				inc = self.curr_function_val[user]
				for msg_no in self.msg_exo.nonzero()[0]:
					user_of_msg = int(self.train[msg_no,0])
					if user_of_msg == user | self.nodes_end[user_of_msg]:
					
						msg_mat = self.get_influence_matrix_for_msg(msg_no)
						inc += self.evaluate( self.covariance[user_of_msg] + msg_mat) - self.curr_function_val[user_of_msg] 
					if inc > max_inc : 
						user_to_choose = user 
						msg_to_choose = msg_no
			if self.msg_end[msg_to_choose]:
				print "a msg which is already endogenious has been selected again as endogenious msg"
			if self.nodes_end[user_to_choose]:
				print "a node which is already endogenious has been selected again as endogenious node"

			return msg_to_choose, user_to_choose
	def update(msg_no = -1 , user = -1 ) : 
		# user
		if user > -1:
			# nodes exo
			# nodes end
			self.nodes_exo[user] = False #.remove(user)
			self.nodes_end[user] = True #.insert(user)		
		if msg_no > -1 : 
			# msg_end
			# msg_exo 
			self.msg_exo[ msg_no ] = False #.remove(msg)
			self.msg_end[ msg_no ] = True #.insert(msg)
			user_of_msg = int( self.train[msg_no, 0])
			# covariance
			self.covariance[user_of_msg] +=  self.get_influence_matrix_for_msg(msg_no)
			# curr_function_val
			self.curr_function_val[user_of_msg] = self.evaluate( self.covariance[ user_of_msg ]) 
	
	def demarkate_process(self, frac_nodes_end, frac_msg_end): 
		#---------------CHANGE----------------------------------
		# create two mat, msg_end, msg_exo
		# nodes_end , nodes_exo
		# frac_nodes_end , frac_msg_end
		# return nodes_end , msg_end
		#-------------------------------------------------
		max_end_user = int(frac_nodes_end*self.num_nodes)
		max_end_msg = int(frac_msg_end * self.num_train)

		self.nodes_end = ma.make_mask(np.zeros(self.num_nodes)) 
		self.nodes_exo = ma.make_mask(np.ones(self.num_nodes))
		self.msg_end = ma.make_mask(np.zeros(self.num_train)) 
		self.msg_exo = ma.make_mask(np.ones(self.num_train)) 

		self.create_influence_matrix() # self.influence_matrix 
		self.create_covariance_matrix() # self.covariance 
		self.create_function_val() # self.curr_function_val 

		while  np.count_nonzero(self.msg_end) < max_end_msg: 
			if np.count_nonzero(self.nodes_end) < max_end_user:
				msg_no , user = self.obtain_most_endogenius_msg_user()
				self.update( msg_no, user)		
			else:
				msg_no = self.obtain_most_endogenius_msg_user(flag_send_msg_only = True)
				self.update( msg_no, user = -1)
		# -----------------------------------------------
		# delete extra  files
		if hasattr( self, 'covariance'):
			del self.covariance
		if hasattr( self, 'influence_matrix'):
			del self.influence_matrix
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
		slant_input_data_obj = slant_input_data( self.nodes, self.edges, self.train[self.msg_end,:] , self.test, flag_evaluate_real = True)
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
	cherrypick_obj.demarkate_process() 
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