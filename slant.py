import os
from myutil import *
from spg_new import spg
from data_preprocess import data_preprocess
import matplotlib.pyplot as plt
from create_synthetic_data import create_synthetic_data
import pickle
from PriorityQueue import PriorityQueue
import math
import numpy.random as rnd
import sys
import numpy as np
from numpy import linalg as LA
# from myutil import myutil
# from scipy import linalg as scipyLA
# from a import graph
# import heapq
# # from collections import deque
# def Grad_f_n_f(self, coef_mat_user, last_coef_val, num_msg_user, msg_time_exp_user,  x):
# 		# f(x) = t2(x) - t1(x)
# 		last_time_train = self.train[-1,1] 
# 		mu = x[0]
# 		b = x[1:]
# 		common_term = np.reciprocal(coef_mat_user.dot(b) * msg_time_exp_user + mu)
# 		del_b_t1 = coef_mat_user.T.dot( common_term *  msg_time_exp_user)
# 		del_mu_t1= np.sum(common_term)
# 		del_b_t2 = (np.exp(self.v*last_time_train)*(last_coef_val) - num_msg_user) / self.v
# 		del_mu_t2= last_time_train
# 		del_t1 = np.concatenate( ([del_mu_t1], del_b_t1))
# 		del_t2 = np.concatenate( ([del_mu_t2], del_b_t2))
# 		# function value computation 
# 		t1 = np.sum( np.log( coef_mat_user.dot(b) * msg_time_exp_user + mu ) )
# 		t2 = mu + (  np.exp(self.v*last_time_train)*(b.dot(last_coef_val)) - b.dot(num_msg_user)) / self.v

# 		grad_f = del_t2 - del_t1 
# 		function_val = t2 - t1  
# 		return grad_f , function_val
class results : 
	def __init__(self, result_val_MSE, result_val_FR, predicted_val, original_val):
		self.result_val_MSE = result_val_MSE 
		self.result_val_FR = result_val_FR 
		self.predicted_val = predicted_val
		self.original_val = original_val
class parameter:
	def __init__( self, mu, alpha, A, B):
		self.alpha = alpha
		self.mu = mu
		self.A = A
		self.B = B


class Grad_n_f:
	def __init__(self, coef_mat_user, last_coef_val_user, num_msg_user, msg_time_exp_user, last_time_train, v):
		self.coef_mat_user = coef_mat_user 
		self.last_coef_val_user = last_coef_val_user
		self.num_msg_user = num_msg_user
		self.msg_time_exp_user = msg_time_exp_user
		self.last_time_train = last_time_train
		self.v = v 
	def f(self,x):
		# function value computation 
		mu = x[0]
		b = x[1:]
		t1 = np.sum( np.log( self.coef_mat_user.dot(b) * self.msg_time_exp_user + mu ) )
		t2 = mu*self.last_time_train - (  np.exp(-self.v*self.last_time_train)*(b.dot(self.last_coef_val_user)) - b.dot(self.num_msg_user)) / self.v	
		return ( t2 - t1 )
		
	def grad_f(self,x):
		mu = x[0]
		b = x[1:]
		common_term = np.reciprocal(self.coef_mat_user.dot(b) * self.msg_time_exp_user + mu)
		del_b_t1 = self.coef_mat_user.T.dot( common_term *  self.msg_time_exp_user)
		del_mu_t1= np.sum(common_term)
		del_b_t2 = -(np.exp(-self.v*self.last_time_train)*(self.last_coef_val_user) - self.num_msg_user) / self.v
		del_mu_t2= self.last_time_train
		del_t1 = np.concatenate( ([del_mu_t1], del_b_t1))
		del_t2 = np.concatenate( ([del_mu_t2], del_b_t2))
		grad_f = del_t2 - del_t1 
		return grad_f 
class slant:
	# check object passing
	def __init__(self,obj, flag_generate_synthetic = False , flag_evaluate_real = False, flag_evaluate_synthetic = False):
		self.num_node = obj.num_node
		self.nodes = obj.nodes
		self.edges=obj.edges

		if (flag_evaluate_real == True ) or (flag_evaluate_synthetic == True):
			self.train= obj.train
			self.test = obj.test

			# check whether to shift user set from 1:N to 0:N-1
			if flag_evaluate_real ==True:
				self.train[:,0] = self.train[:,0] - 1
				self.test[:,0] = self.test[:,0] - 1

			self.num_train= self.train.shape[0]
			self.num_test= self.test.shape[0]

			print "num_train = " + str(self.num_train)
			print "num_test = " + str(self.num_test)
		if flag_generate_synthetic == True: # if we use sentiment polarity we do not need this 
			self.num_sentiment_val = obj.num_sentiment_val 
	
		if flag_evaluate_synthetic == True:
			self.mu_true = obj.mu
			self.alpha_true = obj.alpha
			self.A_true = obj.A
			self.B_true = obj.B

		# self.num_simulation = 1 # 00000 # 
		self.w = 4.0# 
		self.var = .1 # check
		self.v = 2.0 #
		self.lambda_least_square = 1.0 # check this value too
		# self.spectral_nu = 10^(-3)
		self.queue_length = 100
		self.max_iteration_mu_B = 1000

		# self.size_of_function_val_list = 1
		# self.MSE
		# alpha
		# A
		# mu
		# B

	def project_positive(self,v):
		return np.maximum(v,np.zeros(v.shape[0]))

	# def Grad_f_n_f(self, coef_mat_user, last_coef_val, num_msg_user, msg_time_exp_user,  x):
	# 	# f(x) = t2(x) - t1(x)
	# 	last_time_train = self.train[-1,1] 
	# 	mu = x[0]
	# 	b = x[1:]
	# 	common_term = np.reciprocal(coef_mat_user.dot(b) * msg_time_exp_user + mu)
	# 	del_b_t1 = coef_mat_user.T.dot( common_term *  msg_time_exp_user)
	# 	del_mu_t1= np.sum(common_term)
	# 	del_b_t2 = (np.exp(self.v*last_time_train)*(last_coef_val) - num_msg_user) / self.v
	# 	del_mu_t2= last_time_train
	# 	del_t1 = np.concatenate( ([del_mu_t1], del_b_t1))
	# 	del_t2 = np.concatenate( ([del_mu_t2], del_b_t2))
	# 	# function value computation 
	# 	t1 = np.sum( np.log( coef_mat_user.dot(b) * msg_time_exp_user + mu ) )
	# 	t2 = mu + (  np.exp(self.v*last_time_train)*(b.dot(last_coef_val)) - b.dot(num_msg_user)) / self.v

	# 	grad_f = del_t2 - del_t1 
	# 	function_val = t2 - t1  
	# 	return grad_f , function_val
	def find_mu_B(self):
		#-------------------------
		# init_function_val = np.zeros(self.num_node)
		# end_function_val =  np.zeros(self.num_node)
		#-------------------------
		mu=np.zeros(self.num_node)
		B=np.zeros((self.num_node,self.num_node))
		spg_obj = spg() 
		#  gradient calculation
		coef_mat = np.zeros((self.num_train, self.num_node))
		last_coef_val = np.zeros( self.num_node)
		num_msg_users = np.zeros(self.num_node)
		msg_time_exp = np.zeros(self.num_train)
		msg_index = 0 
		for user , time , sentiment  in self.train:
			user = int(user)
			neighbours_with_user  = np.concatenate(([user], np.nonzero(self.edges[user,:])[0] ))
			value = np.exp(self.v * time)
			msg_time_exp[msg_index] = 1/value
			# print user
			last_coef_val[user] +=  value
			coef_mat[msg_index,neighbours_with_user] = last_coef_val[neighbours_with_user]
			num_msg_users[user] = num_msg_users[user] + 1
			msg_index = msg_index + 1 
		
		for user in self.nodes:
			# self.mu[user], self.B[user,] =self.spectral_proj_grad()
			neighbours_with_user  = np.concatenate(([user], np.nonzero(self.edges[user,:])[0] ))
			x0  = np.ones(1+ neighbours_with_user.shape[0])
			# compute parameters************
			user_msg_index = np.where( self.train[:,0]==user )[0]
			# print "-------check-----------"
			# print user_msg_index
			# print "-----------------------"
			# print neighbours_with_user
			coef_mat_user =  coef_mat[user_msg_index.reshape( user_msg_index.shape[0],1),neighbours_with_user.reshape(1, neighbours_with_user.shape[0])]
			msg_time_exp_user = msg_time_exp[user_msg_index]
			last_coef_val_user = last_coef_val[neighbours_with_user]
			num_msg_user=num_msg_users[neighbours_with_user]
			last_time_train = self.train[-1,1]
			#-------------------------------------------
			grad_function_obj = Grad_n_f(coef_mat_user, last_coef_val_user, num_msg_user, msg_time_exp_user, last_time_train, self.v)

			# init_function_val[user] = grad_function_obj.f(x0)
			result= spg_obj.solve( x0, grad_function_obj.f, grad_function_obj.grad_f, self.project_positive, self.queue_length , sys.float_info.epsilon, self.max_iteration_mu_B )
			x = result['bestX']
			# end_function_val[user] = grad_function_obj.f(x)
			mu[user] = x[0]
			B[neighbours_with_user, user] = x[1:]	

		self.mu = mu
		self.B = B
		# plt.plot( init_function_val , 'r')
		# plt.plot( end_function_val , 'b')
		# plt.show()
		# print "likelihood function starts with "+ str( np.sum( init_function_val ) ) + " and ends with " + str( np.sum( end_function_val))

			 

	def solve_least_square(self,A,b):
		
		# print "A"+ str(A.shape)
		# print b.shape

		A_T_b = A.T.dot(b) 
		# print "shape "+ str(A_T_b.shape[0])
		# x=LA.solve( np.matmul(A.T,A)+self.lambda_least_square*np.eye( A_T_b.shape[0] ) , A.T.dot(b))
		mat = np.matmul(A.T,A) + (self.lambda_least_square*np.eye( A_T_b.shape[0] )) 
		# print "rank" + str(LA.matrix_rank(mat))
			# print mat
		# print "size" + str(mat.shape[0]) + "," + str(mat.shape[1])
		# for i in range(12):
		# 	if LA.norm(mat[:,i]) == 0 : 
		# 		print "col zero"
		x=LA.solve( mat , A_T_b)

		# x=LA.solve( self.lambda_least_square*np.eye( A_T_b.shape[0] ) , A.T.dot(b))
		# x=LA.solve( np.matmul(A.T,A), A.T.dot(b))
		# print x.shape
		# return x #**************
		return x[0],x[1:]
	

	def find_alpha_A(self):
		alpha = np.zeros(self.num_node)
		A = np.zeros((self.num_node, self.num_node ))
		index={}
		for user in self.nodes : 
			index[user] = np.where(self.train[:,0]==user)[0]
		for user in self.nodes :  
			num_msg_user = index[user].shape[0]
			if num_msg_user > 0 :
				neighbours = np.nonzero(self.edges[user,:])[0]
				num_nbr = neighbours.shape[0]
				g_user = np.zeros( ( num_msg_user, num_nbr ) )
				msg_user = self.train[index[user],2]
				nbr_no=0
				for nbr in neighbours :
					if index[nbr].shape[0] > 0 :
						user_msg_ind = 0
						time = 0 
						opn = 0 
						index_for_both = np.sort( np.concatenate((index[user], index[nbr])) )
						for ind in index_for_both : 
							user_curr , time_curr , sentiment = self.train[ind,:]
							if user_curr == user:
								opn = opn*np.exp(-self.w*(time_curr - time))
								g_user[user_msg_ind, nbr_no]=opn
								user_msg_ind = user_msg_ind + 1
								if user_msg_ind == num_msg_user:
									break
							else:
								opn = opn*np.exp(-self.w*(time_curr - time))+sentiment
							time = time_curr
					nbr_no = nbr_no + 1
				g_user = np.concatenate(( np.ones((msg_user.shape[0],1)) , g_user  ), axis = 1 )
				alpha[user], A[neighbours, user] = self.solve_least_square( g_user, msg_user )
		self.alpha = alpha
		self.A = A



		# print " diff of alpha "
		# print LA.norm(self.alpha - self.alpha_true)
		# print " diff of A "
		# print LA.norm( self.A - self.A_true)
	def generate_parameters(self):
		self.mu = rnd.uniform(low = 0 , high = .05 , size = self.num_node)
		self.alpha = rnd.uniform( low = 0 , high = self.num_sentiment_val-1 , size = self.num_node )
		self.A = np.zeros(( self.num_node , self.num_node ))
		self.B = np.zeros(( self.num_node , self.num_node ))
		for user in self.nodes:
			nbr = np.nonzero(self.edges[user,:])[0]
			# print "___user "+ str(user)+"_________"
			# print nbr
			# print "__________A_________"
			self.A[user ,  nbr ] = rnd.uniform( low = -1 , high = 1 , size = nbr.shape[0] ) 
			# print self.A[user , :]
			# print "_____________________"
			self.B[user , np.concatenate(([user], nbr ))] = rnd.uniform( size = nbr.shape[0] + 1  )
			# print self.B[user , :]
	def estimate_param(self):
		# pass
		# estimate parameters
		# print "inside"
		# self.mu,self.B = self.find_mu_B()
		self.find_mu_B()
		self.find_alpha_A()
		# print self.mu
		# print self.alpha
		# print "____________E__________________________________________--______"
		# print self.edges
		# print "----------A-------------------------------------------------------"
		# print self.A
		# print "___________B___________________________________________________--------___------"
		# print self.B
		# print "----------nodes-------------"
		# print self.nodes
	def check_param_diff(self):
		print "to be defined"
	def set_parameter(self, obj):
		self.mu = obj.mu
		self.alpha = obj.alpha 
		self.A = obj.A
		self.B = obj.B
	
		
	def predict(self, num_simulation, time_span_input):
		# test is a dictionary
		# test['user']=[set of msg posted by that user]
		# each user is a key 
		# each user has a list of messages attached to it.
		# for each message ,
		# sample a set of msg first
		# predict the msg of that user at that time
		# save the prediction in a dictionary called prediction 
		# return a set of predicted msg
		# add a loop here to run the following simulation repeated times
		# v1 = np.array([1,2,2])
		# v2 = np.array([4,5,6])
		# print self.get_MSE(v1,v2)

		print "prediction start"
		predict_test = np.zeros( self.test.shape[0] )
		self.update_opn_int_history(-1)
		# print " train 1 : 20 time "
		# print self.train[0:20,1]
		# print "----------------------------------"
		filename = "current_status"
		if os.path.exists(filename):
			os.remove(filename)
		for msg_index in range(self.test.shape[0]): 
		
			with open( filename ,'a+') as f :
				f.write(str(msg_index) + " th msg complete\n")
			print "----------- test msg number " + str(msg_index)
			# self.predict_single_instance( self.test[msg_index], num_simulation, time_span_input) # check indexing # change
			predict_test[msg_index] = self.predict_single_instance( self.test[msg_index], num_simulation, time_span_input) 

		MSE_loss = get_MSE(predict_test, self.test[:,2])
		FR_loss = get_FR(predict_test, self.test[:,2])
		return MSE_loss, FR_loss, predict_test

		#----------------------------to do -------------------------------------
		# check get mse 
		# solve why intensity = 0 
	def update_opn_int_history(self, curr_time): 
		# for all user, update their opn and int using msg from history upto current time and saves those values in curr_opn, curr_int
		# Also that case is not covered when test - time span exceeds last msg of train set 
		if curr_time == -1 :
			# indicates it is called for initialization
			self.curr_msg_index=-1
			self.curr_opn = self.alpha
			self.curr_int = self.mu
			return


		time_array = self.train[:,1]
		if curr_time > self.train[-1,1]:
			if curr_time > self.test[0,1]:
				print(" we need test samples to update opinion and intensity . But that case is not covered")
		max_msg_index = np.count_nonzero( time_array < curr_time) -1
		# print "-------------inside update opn int --------------------"
		# print "start msg index = " + str(self.curr_msg_index+1)
		# print "end msg index = " + str(max_msg_index)
		for msg_index in range( self.curr_msg_index+1  , max_msg_index+1 , 1): #check , perhaps terminal conditions are incorrect
			# print msg_index
			user, time, sentiment = self.train[msg_index,:]
			user = int(user)
			if msg_index == 0 :
				time_diff = time
			else:
				time_diff = time - self.train[msg_index-1,1]
			self.curr_opn = self.alpha + (self.curr_opn - self.alpha) * np.exp( - self.w * time_diff) + self.A[user]*sentiment # define time diff 
			self.curr_int = self.mu + ( self.curr_int - self.mu) * np.exp( - self.v * time_diff) + self.B[user] # check it with abirda whether my curr msg will affect my intensity , perhaps it will
		self.curr_msg_index = max_msg_index
		# if np.count_nonzero(self.curr_int) == 0:
			# print np.count_nonzero(self.curr_int)
		# print "number of  node " + str(self.num_node) + " with number of msg in train set = " + str(self.num_train)
		# print "number of nodes with non zero intensity " + str(np.count_nonzero(self.curr_int))
	def predict_single_instance( self, msg, num_simulation, time_span_input):
		user, time, sentiment = msg 
		# print " msg = " + str(user) + " " + str(time) + " " + str(sentiment)
		user = int(user)
		# get initial opn from history
		start_time_sampling = time - time_span_input
		# print " sampling start time " + str(start_time_sampling)
		self.update_opn_int_history( start_time_sampling)
		# self.plot_user_vs_msg()
		prediction_array = np.zeros( num_simulation )
		for simulation_no in range(num_simulation):
			# sample using opinions obtained in previous step
			#-------------------------------------------------------------------DELETE -------------------------------------
			# self.curr_int = rnd.rand(self.num_node)*4
			# self.curr_opn = rnd.uniform(-1,1, self.num_node)

			#---------------------------------------------------------------------
			msg_set, last_opn_update, last_int_update =  self.simulate_events(time_span_input , self.curr_int, self.curr_opn,  self.A, self.B)
			# find_num_msg_plot( msg_set, self.num_node, self.curr_int)
			# all three above variable store times assuming start time as 0 whereas start time is actually "time"
			# therefore before using them they must be corrected
			msg_set[:,1] += start_time_sampling
			last_opn_update[user,0] += start_time_sampling
			prediction_array[simulation_no] = self.predict_from_events( self.alpha[user], self.A[:,user] , last_opn_update[user,:],  msg_set, user , time ) # or perhaps next one 
			
			# prediction_array[simulation_no] = myutil.find_opn_markov( check_the_arguements ) # or perhaps next one 
		# return mean of all predictions 
		return np.mean( prediction_array)
	def simulate_events(self, time_span , mu, alpha, A, B, flag_check_num_msg = False, return_only_opinion_updates= False):

		#---------------------self variables used in this module ---------
		# edges
		# w
		# v 
		#-----------------------------------------------------------------
		
		time_init = np.zeros((self.num_node,1))
		opn_update = np.concatenate((time_init, alpha.reshape(self.num_node , 1 )), axis=1)
		int_update =  np.concatenate((time_init, mu.reshape( self.num_node , 1 )), axis=1)
		
		msg_set = []

		tQ=np.zeros(self.num_node)
		for user in self.nodes:
			# if mu[user] == 0 :
			# 	print "initial intensity  = zero "
			tQ[user] = self.sample_event( mu[user] , 0 , user, time_span ) 
			# tQ[user] = rnd.uniform(0,T)
		# Q=PriorityQueue(tQ) # set a chcek on it
		# print "----------------------------------------"
		# print "sample event starts"
		t_new = 0
		#--------------------------------------------------
		if flag_check_num_msg==True:
			num_msg = 0
		#--------------------------------------------------
		while t_new < time_span:
			u = np.argmin(tQ)
			t_new = tQ[u]
			tQ[u] = float('inf')
			# t_new,u=Q.extract_prior()# do not we need to put back t_new,u * what is this t_new > T 
			# u = int(u)

			# print " extracted user " + str(u) + "---------------time : " + str(t_new)
			# t_old=opn_update_time[u]
			# x_old=opn_update_val[u]
			t_old,x_old = opn_update[u,:]
			x_new=alpha[u]+(x_old-alpha[u])*np.exp(-self.w*(t_new-t_old))
			# opn_update_time[u]=t_new
			# opn_update_val[u]=x_new
			opn_update[u,:]=np.array([t_new,x_new])
			m = rnd.normal(x_new,self.var)
			msg_set.append(np.array([u,t_new,m]))
			if flag_check_num_msg == True:
				num_msg = num_msg + 1 
				if num_msg > max_msg :
					break
			# update neighbours
			for nbr in np.nonzero(self.edges[u,:])[0]:
				# print " ------------for nbr " + str(nbr) + "-------------------------"
				# change above 
				t_old,lda_old = int_update[nbr]
				lda_new = mu[nbr]+(lda_old-mu[nbr])*np.exp(-self.v*(t_new-t_old))+B[u,nbr]# use sparse matrix
				int_update[nbr,:]=np.array([t_new,lda_new])
				t_old,x_old=opn_update[nbr]
				x_new = alpha[nbr] + ( x_old - alpha[nbr] )*np.exp(-self.w*(t_new-t_old)) + A[u,nbr]*m
				opn_update[nbr]=np.array([t_new,x_new])

				# print " updated int " + str(lda_new) + " ------------ updated opinion -----" + str(x_new)
				t_nbr=self.sample_event(lda_new,t_new,nbr, time_span )
				# print " update next event time of " + str( nbr ) + "  as " + str(t_nbr)
				tQ[nbr]=t_nbr
				# Q.update_key(t_nbr,nbr) 
			# Q.print_elements()
			
		if return_only_opinion_updates == True:
			return opn_update
		else:
			return np.array(msg_set) , opn_update, int_update
	def sample_event(self,lda_init,t_init,user, T ): # to be checked
		lda_old=lda_init
		t_new= t_init
		 
		
		# print "------------------------"
		# print "start tm "+str(t_init) + " --- int --- " + str(lda_init)
		# print "------------start--------"
		# itr = 0 
		while t_new< T : 
			u=rnd.uniform(0,1)
			# if lda_old == 0:
			# 	print "itr no " + str(itr)
			t_new -= math.log(u)/lda_old
			# print "new time ------ " + str(t_new)
			lda_new = self.mu[user] + (lda_init-self.mu[user])*np.exp(-self.v*(t_new-t_init))
			# print "new int  ------- " + str(lda_new)
			d = rnd.uniform(0,1)
			# print "d*upper_lda : " + str(d*lda_upper)
			if d*lda_old < lda_new  :
				break
			else:
				lda_old = lda_new
			# itr += 1
				

		return t_new # T also could have been returned

	def predict_from_events( self, alpha, A , last_opn_update,  msg_set, user , time ): # confirm from abir da whether to send alpha or curr opn variable of self 
		time_array = msg_set[:,1]
		time_last, opn  = last_opn_update 
		
		for user_curr, time_curr, sentiment in msg_set[np.logical_and((time_array > time_last ), ( time_array < time)) ,:]: 
			if self.edges[user,int(user_curr)]>0 : 
				time_diff = time_curr - time_last
				opn = alpha + (opn - alpha )*np.exp(-self.w*time_diff)+A[int(user_curr)]*sentiment
				time_last = time_curr
		time_diff = time - time_last
		opn = alpha + (opn - alpha )*np.exp(-self.w*time_diff)
		return opn
	def check_validity_of_param(self):
		if np.count_nonzero(self.mu) == 0 :
			print "all initial intensity are zero"
		if np.count_nonzero(self.alpha) == 0 :
			print "all initial opinion are zero"
		if np.count_nonzero(self.A) == 0 :
			print "all initial opinion influence are zero"
		if np.count_nonzero(self.B) == 0 :
			print "all initial intensity influence are zero"
	def plot_user_vs_msg(self):
		num_msg_per_user = np.zeros(self.num_node)
		for user in self.nodes:
			num_msg_per_user[user] = np.count_nonzero(self.train[:,0]==user)
		plt.plot( num_msg_per_user, 'r')
		# plt.plot( self.curr_int, 'b')

		plt.plot( self.mu, 'b')
		plt.show()
def load_results(result_file):
	result_obj = load_data(result_file)
	print "MSE  " + str(result_obj.result_val_MSE)
	print "FR  " + str(result_obj.result_val_FR)
	index  = np.arange(200)
	plt.plot(result_obj.predicted_val[index], "r")
	plt.plot(result_obj.original_val[index], "b")
	plt.show()


# check util functionalities---------------------------------------------------------------------------------

	
#----------------------------------------------------send these to util------------------------------------
# def save_parameter( mu, alpha, A, B , filename ):

def main():

	flag_generate_dataset_by_simulation =  False
	flag_train_evaluate_real_data =  True # False
	flag_train_evaluate_synthetic_data = False 
	flag_show_results =  False # True
	# flag_train = False
	# flag_evaluate = False
	# flag_test_estimation_process = True # False
	
	# load data 
	#------------------------------------------------------------------------
	# modify each predict by adding num_simulation
	#------------------------------------------------------------------------

	if flag_train_evaluate_real_data==True:
		
		path = '../Cherrypick_others/Data_opn_dyn_python/'
		file = 'jaya_verdict_10ALLXContainedOpinionX'
		filename = path + file + '.obj'
	
			# input file has an object with following components
			# nodes
			# edges
			# train ( np array  of msg )
			# test ( np array  of msg )
		##----------------------Initialize
		obj = load_data(filename)
		slant_obj = slant( obj , flag_evaluate_real = True)  
		# slant_obj.check()
		## -----------------------------------------------estimate parameters
		flag_load_param =False 
		if flag_load_param :
			param_obj=load_data(path+file+'.param')
			slant_obj.set_parameter( param_obj)

		else:
			slant_obj.estimate_param( ) # estimate param must be checked , after checking predict 
		flag_save_param = True # False
		if flag_save_param  == True:
			param_obj = parameter(  slant_obj.mu, slant_obj.alpha, slant_obj.A, slant_obj.B)
			save( param_obj, path+ file + '.param')
		print "estimation over"
		# print "fraction of test = " + str( slant_obj.test.shape[0]/float(slant_obj.train.shape[0]))
		# plt.plot(slant_obj.train[1:500,1])
		# plt.show()
		# ## --------------------------------------------------Predict
		# slant_obj.check_validity_of_param()
		# slant_obj.plot_user_vs_msg()
		#-----------------------------------------------------------------
		# slant_obj.test  = slant_obj.test[0,:]
		# slant_obj.predict( num_simulation = 1, time_span_input = 4 ) # change no of simulation
		#-------------------------------------------C  H  A  N  G  E --------------------------------------------------
		result_val_MSE, result_val_FR, predicted_val = slant_obj.predict( num_simulation = 1 , time_span_input = 4 )
		print "prediction over"
		print "MSE" + str(result_val_MSE)
		print "FR" + str(result_val_FR)
		# # # result_val = rnd.rand(1)
		# # # predicted_val = rnd.rand(slant_obj.test.shape[0])
		
		result_obj = results( result_val_MSE, result_val_FR, predicted_val, slant_obj.test[:,2])
		result_file = path + file + ".res"
		save( result_obj, result_file)
	if flag_train_evaluate_synthetic_data ==True:
		input_file = 'synthetic_dataset_1'
		slant_obj = slant( load_data(input_file), flag_evaluate_synthetic = True)
		# checking parameter estimation process
		slant_obj.estimate_param(  )
		slant_obj.check_param_diff() # to be estimated
		slant_obj.plot_user_vs_msg()
		result = slant_obj.predict()

	if flag_generate_dataset_by_simulation == True:
		# ##################################################
		# Input : Graph
		# Output : set of messages, set to train
		# Output is saved as output_file
		input_file = 'synthetic_graph'
		load_data(input_file)
		slant_obj = slant( load_data(input_file), flag_generate_synthetic = True)
		slant_obj.generate_parameters() # -------------------------Generate mu , alpha , A , B
		T = 100
		opn_updates , int_updates , msg_set = slant_obj.simulate_events(T, return_only_opinion_updates = False, flag_check_num_msg = True, max_msg = 100)
		slant_obj.set_msg(flag_only_train = True, msg_set = msg_set) # set both train and test 
		# slant_obj.flag_synthetic_data = True
		output_file = 'synthetic_dataset_1'
		save(slant_obj, output_file)
	if flag_show_results :

		path = '../Cherrypick_others/Data_opn_dyn_python/'
		file = 'jaya_verdict_10ALLXContainedOpinionX'
		filename = path + file + '.res'

		load_results(filename)









if __name__=="__main__":
	main()

		
# def find_alpha_A(self):
	# 	# maintain msg counter for each user
	# 	# map users to 0 to nusers
	# 	ind_4_V=np.zeros(self.nuser)
	# 	tau={}
	# 	for u in range(nuser):
	# 		tau[u]=[]
	# 	for t,u,m in H: #
	# 		tau[u].append([t,u,m])
	# 		ind_4_v[u]=ind_4_v[u]+1
	# 	for u in range(nuser):
	# 		i=0
	# 		S=tau[u]
	# 		for nbr in graph[u]:
	# 			S=self.merge(S,tau[nbr])#
	# 			x_last=0
	# 			t_last=0
	# 			m_no=0
	# 			for t,v,m in S:
	# 				if v==u:
	# 					x=x_last*math.exp(-w*(t-t_last))
	# 					g(m_no,v)=x
	# 					y(m_no)=m
	# 					m_no=m_no+1
	# 				else:
	# 					x=x_last*math.exp(-w*(t-t_last))
	# 				t_last=t
	# 				x_last=x
	# 		alpha[u],A[u]=self.solve_least_square(g,y,lda)
	# 	# nmsg
	# 	# M=np.array(nuser,nmsg.max(),3)
	# 	# for u in self.graph.vertices:
	# 	# 	M[u,:,:1]=np.asarray([train[i,1:] if train[i,0]==u for i in self.ntrain])
	# 	# 	M[u,:,2]=np.multiply(M[u,:,0],np.exp(w*M[u,:,1]))
	# 	# alpha=np.array(nuser)
	# 	# A=[]
	# 	# for u in self.graph.vertices:
	# 	# 	M_hat=np.array(nmsg,len(self.edges[u])+1)
	# 	# 	arg1=M_hat.transpose().dot(M_hat)
	# 	# 	arg2=M_hat.transpose().dot(M[u,:,0])
	# 	# 	res = LA.solve(arg1,arg2)
	# 	# 	alpha[u]=res[0]
	# 	# 	A.append(list(res[1:]))
	# 	# 	alpha[u],A[u]=self.solve_least_square(A,b)
	# 	return alpha,A








	#********************************************************
	#----------- Mu and B -----------------------------------
	#########################################################
	# def find_mu_B(self):
	# 	mu=np.ones(self.num_node)
	# 	B=np.zeros((self.num_node,self.num_node))

	# 	#  gradient calculation
	# 	coef_mat = np.zeros((self.num_train, self.num_node))
	# 	last_coef_val = np.zeros( self.num_node)
	# 	num_msg_users = np.zeros(self.num_node)
	# 	msg_time_exp = np.zeros(self.num_train)
	# 	msg_index = 0 
	# 	for user , time , sentiment  in self.train:
	# 		user = int(user)
	# 		neighbours_with_user  = np.concatenate(([user], np.nonzero(self.edges[user,:])[0] ))
	# 		value = np.exp(-self.v * time)
	# 		msg_time_exp[msg_index] = 1/value
	# 		# print user
	# 		last_coef_val[user] = last_coef_val[user] + value
	# 		coef_mat[msg_index,neighbours_with_user] = last_coef_val[neighbours_with_user]
	# 		num_msg_users[user] = num_msg_users[user] + 1
	# 		msg_index = msg_index + 1 
		
	# 	for user in self.nodes:
	# 		# self.mu[user], self.B[user,] =self.spectral_proj_grad()
	# 		neighbours_with_user  = np.concatenate(([user], np.nonzero(self.edges[user,:])[0] ))
	# 		x=np.ones(1+ neighbours_with_user.shape[0])
	# 		# func_val_list=[]
	# 		d = np.ones( x.shape[0] ) 
	# 		H = np.eye( x.shape[0] ) 
	# 		alpha_bb = min([.0001 +.5*rnd.uniform(0,1) , 1 ])
	# 		alpha_min = .0001
	# 		# compute parameters***************  
	# 		user_msg_index = np.where( self.train[:,0]==user )[0]
	# 		coef_mat_user =  coef_mat[user_msg_index,neighbours_with_user]
	# 		msg_time_exp_user = msg_time_exp[user_msg_index]
	# 		last_coef_val_user = last_coef_val[neighbours_with_user]
	# 		num_msg_user=num_msg_users[neighbours_with_user]
	# 		# user_mask = self.edges[user,:]
	# 		# user_mask[user] = 1
		
			
	# 		while LA.norm(d) > sys.float_info.epsilon:

	# 			grad_f , likelihood_val = self.Grad_f_n_f(coef_mat_user, last_coef_val_user, num_msg_user ,  msg_time_exp_user,  x)  # user_mask,
	# 			alpha_bar=min([alpha_min,max(alpha_bb, alpha_min)]) 
	# 			# grad_f = np.ones(self.num_node+1) #**
	# 		# 	# alpha
	# 			d=self.project_positive(x-alpha_bar*grad_f)-x #
	# 		# 	# func_val_list.append(self.f(x)) #
	# 		# 	# if len(func_val_list) > self.size_of_function_val_list: 
	# 		# 		# func_val_list.pop(0)
	# 		# 	# max_func_val=max(func_val_list) 
	# 			alpha = 1
	# 			while ( alpha*d.dot(grad_f) + np.power(alpha,2)*(d.dot(H.dot(d))) > spectral_nu*alpha*(grad_f.dot(d)) ) & (alpha > 0)  : # lhs # write B as H 
	# 				alpha = alpha - .1
	# 				print "alpha is " + str(alpha)
	# 			print alpha
				
	# 			s = alpha * d
	# 			x = x + s


	# 			y = H.dot(d)
	# 			alpha_bb = y.dot(y) / s.dot(y) 
	# 			# Bk=Bk-Bk*(sk*sk')*Bk/(sk'*Bk*sk)+yk*yk'/(yk'*sk);
	# 			H_s = H.dot(s).reshape(H.dot(s).shape[0],1)
	# 			y_2dim = y.reshape(y.shape[0],1)
	# 			H = H - s.dot(H.dot(s)) * np.matmul( H_s , H_s.T ) + np.matmul( y_2dim , y_2dim.T )/ y.dot(s)
				
	# 			break # to be deleted
	# 		mu[user] = x[0]
	# 		B[user, neighbours_with_user] = x[1:]	

	# 	self.mu = mu
	# 	self.B = B
	# 


	# def analytical_opinion_forecast_poisson(self,train, test): # analytical using poisson
	# 	# create nbr[u]
	# 	time = train[:,2] - t0 
	# 	temp = np.exp(-time*w).dot(train[:,1])
	# 	sentiment =np.array(nuser)
	# 	for u in range(nuser):
	# 		sentiment[u]=sum(np.asarray([temp[i] if train[i,0]==u for i in range(ntrain)]))
		
	# 	x_t0=self.alpha[u]+A[u,self.graph.edges[u]].dot(sentiment[self.graph.edges[u]])
	# 	mat_arg=self.A.dot(np.diag(self.mu))-w*np.eye(nuser)
	# 	inv_mat = self.inverse_mat(mat_arg)
	# 	res=np.zeros(ntest)
	# 	for m in range(ntest):
	# 		exp_mat = scipyLA.expm(self.delta_t*mat_arg)
	# 		sub_term2 = (exp_mat-eye(nuser))*self.alpha
	# 		res[m] = exp_mat[u]*x_t0+w[u]*inv_mat[u]*sub_term2
	# 	return res	




	#-------------------------------

	# set ready estimate of all
	# once done , run estimate files 
	# change simulation 
	# run 
	# vpn
	# run
	# server acct
	# check data nodes

	#---------------------------------------------------------------------------------------------------------------
	#------------Predict by simulation part has been changed from June 21 ------------------------------------------
	#------------Old version is saved here -------------------------------------------------------------------------

	# def get_FR(self,s,t):
	# 	# print float(s.shape[0] - np.count_nonzero(np.sign(s) + np.sign(t))) / s.shape[0]
	# 	return float(s.shape[0] - np.count_nonzero( np.sign(s) + np.sign(t)))/s.shape[0]
	# def get_MSE(self,s,t):
	# 	return np.mean((s-t)**2)
		
	# def predict(self):
	# 	# test is a dictionary
	# 	# test['user']=[set of msg posted by that user]
	# 	# each user is a key 
	# 	# each user has a list of messages attached to it.
	# 	# for each message ,
	# 	# sample a set of msg first
	# 	# predict the msg of that user at that time
	# 	# save the prediction in a dictionary called prediction 
	# 	# return a set of predicted msg
	# 	# add a loop here to run the following simulation repeated times
	# 	# v1 = np.array([1,2,2])
	# 	# v2 = np.array([4,5,6])
	# 	# print self.get_MSE(v1,v2)



	# 	self.MSE = np.zeros(self.num_simulation)
	# 	for simulation_no in range(self.num_simulation):
	# 		# predict_test =  rnd.uniform( low = 0 , high = self.num_sentiment_val-1  , size = self.num_test )# 
	# 		predict_test =  self.predict_by_simulation() 
	# 		self.MSE[simulation_no] = self.get_MSE(predict_test, self.test[:,2]) # define performance
	# 		# self.performance.FR = self.get_FR(predict_test, test[:,1]) 
	# 	# print np.mean(self.MSE)
	# 	return np.mean(self.MSE)
	# def predict_by_simulation(self):
	# 	t_old= self.train[-1,1]
	# 	# discuss the first case
	# 	predict_test = np.zeros(self.num_test)
	# 	msg_no = 0 
	# 	# for user, time, sentiment in self.test:
	# 	for user, time, sentiment in self.test:
	# 		user = int(user)
	# 		del_t = time-t_old 
	# 		# print user
	# 		# print del_t
	# 		opn_update = self.simulate_events(del_t, return_only_opinion_updates = True) 
	# 		# ----------------------------
	# 		# Could be changed also to have posted msg
	# 		#-----------------------------
	# 		# may send user opn update also # add t_old with time array #************
	# 		# self.simulate_events(del_t, return_only_opinion_updates = True) # may send user opn update also # add t_old with time array
	# 		#-------------------------------------------------------------------
	# 		# opn_update = np.zeros((self.num_node , 2))
	# 		# opn_update[:,0] =  rnd.uniform( low = 0 , high = del_t, size = self.num_node )
	# 		# opn_update[:,1] = rnd.uniform( low = 0 , high = self.num_sentiment_val-1 , size = self.num_node )
	# 		# #---------------------------------------------------------------------
	# 		# print opn_update
	# 		# predict_test[m_no] = self.predict_from_msg_set( user, t_new, msg_set)
	# 		time_diff,opn_last = opn_update[user,:]
	# 		predict_test[msg_no] = self.find_opn_markov(opn_last, del_t - time_diff, self.alpha[user], self.w)
	# 		t_old = time
	# 		msg_no = msg_no + 1 
	# 		# break #--------------------------------------------------------------------------
	# 	return predict_test
	

	# def find_opn_markov(self, opn_last , del_t, alpha, w):
	# 	# print "-----------------------------------------------------------------------------"
	# 	# print opn_last 
	# 	# print del_t 
	# 	# print alpha 
	# 	# print w 
	# 	# print w * del_t
	# 	# print np.exp(- w * del_t) 
	# 	# print (opn_last - alpha)*np.exp(- w * del_t) 
	# 	# print alpha + (opn_last - alpha)*np.exp(- w * del_t) 
	# 	return alpha + (opn_last - alpha)*np.exp(- w * del_t) 
	# def simulate_events(self,T, return_only_opinion_updates = False, flag_check_num_msg = False, max_msg = 0):#
	# 	# test message set , parameters learnt from train , T  , graph ( number of node and adj list )
	# 	# sample events 
	# 	# sample events for each user
	# 	# until we reach T , we generate min t , generate corresponding event ,  update all neighbours 
	# 	# predict message sentiment for each msg in test set
	# 	# return prediction 


	# 	#________________________checking sample events --------------------------------------------

	# 	# self.sample_event(lda_init = .3, t_init =1 , v =1 , T = 5)

	# 	# ------------------------------------------------------------------------------------------
		
	# 	time_init = np.zeros((self.num_node,1))
	# 	opn_update = np.concatenate((time_init, self.alpha.reshape(self.num_node , 1 )), axis=1)
	# 	int_update =  np.concatenate((time_init, self.mu.reshape( self.num_node , 1 )), axis=1)
		
	# 	msg_set = []

	# 	tQ=np.zeros(self.num_node)
	# 	for user in self.nodes:
	# 		tQ[user] = self.sample_event( self.mu[user] , 0 , user, T ) 
	# 		# tQ[user] = rnd.uniform(0,T)
	# 	Q=PriorityQueue(tQ)
	# 	# print "----------------------------------------"
	# 	# print "sample event starts"
	# 	t_new = 0
	# 	if flag_check_num_msg==True:
	# 		num_msg = 0
	# 	while t_new < T:

	# 		t_new,u=Q.extract_prior()# do not we need o put back t_new,u 
	# 		u = int(u)

	# 		# print " extracted user " + str(u) + "---------------time : " + str(t_new)
	# 		# t_old=opn_update_time[u]
	# 		# x_old=opn_update_val[u]
	# 		[t_old,x_old] = opn_update[u,:]
	# 		x_new=self.alpha[u]+(x_old-self.alpha[u])*np.exp(-self.w*(t_new-t_old))
	# 		# opn_update_time[u]=t_new
	# 		# opn_update_val[u]=x_new
	# 		opn_update[u,:]=np.array([t_new,x_new])
	# 		m = rnd.normal(x_new,self.var)
	# 		msg_set.append(np.array([u,t_new,m]))
	# 		if flag_check_num_msg == True:
	# 			num_msg = num_msg + 1 
	# 			if num_msg > max_msg :
	# 				break
	# 		# update neighbours
	# 		for nbr in np.nonzero(self.edges[u,:])[0]:
	# 			# print " ------------for nbr " + str(nbr) + "-------------------------"
	# 			# change above 
	# 			[t_old,lda_old] = int_update[nbr,:]
	# 			lda_new = self.mu[nbr]+(lda_old-self.mu[nbr])*np.exp(-self.v*(t_new-t_old))+self.B[u,nbr]# use sparse matrix
	# 			int_update[nbr,:]=np.array([t_new,lda_new])
	# 			t_old,x_old=opn_update[nbr,:]
	# 			x_new=self.alpha[nbr]+(x_old-self.alpha[nbr])*np.exp(-self.w*(t_new-t_old))+self.A[u,nbr]*m
	# 			opn_update[nbr,:]=np.array([t_new,x_new])

	# 			# print " updated int " + str(lda_new) + " ------------ updated opinion -----" + str(x_new)
	# 			t_nbr=self.sample_event(lda_new,t_new,nbr, T )
	# 			# print " update next event time of " + str( nbr ) + "  as " + str(t_nbr)

	# 			Q.update_key(t_nbr,nbr) 
	# 		# Q.print_elements()
			
	# 	if return_only_opinion_updates == True:
	# 		return opn_update
	# 	else:
	# 		return opn_update, int_update, np.array(msg_set) 
	# def sample_event(self,lda_init,t_init,v, T ): # to be checked
	# 	lda_upper=lda_init
	# 	t_new = t_init
		 
		
	# 	# print "------------------------"
	# 	# print "start tm "+str(t_init) + " --- int --- " + str(lda_init)
	# 	# print "------------start--------"
	# 	while t_new < T : #* get T 
	# 		u=rnd.uniform(0,1)
	# 		t_new = t_new - math.log(u)/lda_upper
	# 		# print "new time ------ " + str(t_new)
	# 		lda_new = self.mu[v] + (lda_init-self.mu[v])*np.exp(-self.v*(t_new-t_init))
	# 		# print "new int  ------- " + str(lda_new)
	# 		d = rnd.uniform(0.9,1)
	# 		# print "d*upper_lda : " + str(d*lda_upper)
	# 		if d*lda_upper < lda_new  :
	# 			break
	# 		else:
	# 			lda_upper = lda_new
	# 	return t_new
	# def set_msg(self, flag_only_train = True, msg_set = [], train_fraction = 0):
	# 	if flag_only_train == True:
	# 		self.train = msg_set
	# 		self.num_train = self.train.shape[0]
	# 	else:
	# 		num_msg = msg_set.shape[0]	
	# 		num_tr = int( num_msg * train_fraction )
	# 		self.train = msg_set[:num_tr, :]
	# 		self.test = msg_set[num_tr : ,:]
	# 		self.num_train = self.train.shape[0]
	# 		self.num_test = self.test.shape[0]
	# 		del msg_set
	# # def check_user_start_index(self):
	# # 	# nodes in msg
	# # 	nodes = np.unique(  self.train[:,0])
	# # 	node_min = np.amin(nodes)
	# # 	node_max = np.amax( nodes)
	# # 	node_range_msg = node_max - node_min + 1 
	# # 	if node_range_msg == self.num_node:
	# # 		print "end nodes appear in msg"
	# # 		if node_min == 1:
	# # 			print " min node is 1 , therefore shifting users"
	# # 			return True
	# # 		else:
	# # 			return False 
	# # 	else:
	# # 		print "Not all nodes appeared in msg"
	# # 		if node_min == 1:
	# # 			print "node min is 1 "
	# # 			print "please check further on this dataset"
	# # 			return True
	# # 	return False
	# 	# print nodes.shape[0]
	# 	# print "-----"
	# 	# print self.num_node
	# 	# print "-----"
	# 	# print np.amax( nodes)
	# 	# print "  -- min -- "
	# 	# print np.amin(nodes)
	# def predict_by_simulation(self):
	# 	t_old= self.train[-1,1]
	# 	# discuss the first case
	# 	predict_test = np.zeros(self.num_test)
	# 	msg_no = 0 
	# 	# for user, time, sentiment in self.test:
	# 	for user, time, sentiment in self.test:
	# 		user = int(user)
	# 		del_t = time-t_old 
	# 		# print user
	# 		# print del_t
	# 		opn_update = self.simulate_events(del_t, return_only_opinion_updates = True) 
	# 		# ----------------------------
	# 		# Could be changed also to have posted msg
	# 		#-----------------------------
	# 		# may send user opn update also # add t_old with time array #************
	# 		# self.simulate_events(del_t, return_only_opinion_updates = True) # may send user opn update also # add t_old with time array
	# 		#-------------------------------------------------------------------
	# 		# opn_update = np.zeros((self.num_node , 2))
	# 		# opn_update[:,0] =  rnd.uniform( low = 0 , high = del_t, size = self.num_node )
	# 		# opn_update[:,1] = rnd.uniform( low = 0 , high = self.num_sentiment_val-1 , size = self.num_node )
	# 		# #---------------------------------------------------------------------
	# 		# print opn_update
	# 		# predict_test[m_no] = self.predict_from_msg_set( user, t_new, msg_set)
	# 		time_diff,opn_last = opn_update[user,:]
	# 		predict_test[msg_no] = self.find_opn_markov(opn_last, del_t - time_diff, self.alpha[user], self.w)
	# 		t_old = time
	# 		msg_no = msg_no + 1 
	# 		# break #--------------------------------------------------------------------------
	# 	return predict_test
	
	# -------------------------------Sampling modules-------------------------------------------------------------------------------

	# def find_opn_markov(self, opn_last , del_t, alpha, w):
	# 	# print "-----------------------------------------------------------------------------"
	# 	# print opn_last 
	# 	# print del_t 
	# 	# print alpha 
	# 	# print w 
	# 	# print w * del_t
	# 	# print np.exp(- w * del_t) 
	# 	# print (opn_last - alpha)*np.exp(- w * del_t) 
	# 	# print alpha + (opn_last - alpha)*np.exp(- w * del_t) 
	# 	return alpha + (opn_last - alpha)*np.exp(- w * del_t) 
	# def simulate_events(self,T, return_only_opinion_updates = False, flag_check_num_msg = False, max_msg = 0):#
	# 	# test message set , parameters learnt from train , T  , graph ( number of node and adj list )
	# 	# sample events 
	# 	# sample events for each user
	# 	# until we reach T , we generate min t , generate corresponding event ,  update all neighbours 
	# 	# predict message sentiment for each msg in test set
	# 	# return prediction 


	# 	#________________________checking sample events --------------------------------------------

	# 	# self.sample_event(lda_init = .3, t_init =1 , v =1 , T = 5)

	# 	# ------------------------------------------------------------------------------------------
		
	# 	time_init = np.zeros((self.num_node,1))
	# 	opn_update = np.concatenate((time_init, self.alpha.reshape(self.num_node , 1 )), axis=1)
	# 	int_update =  np.concatenate((time_init, self.mu.reshape( self.num_node , 1 )), axis=1)
		
	# 	msg_set = []

	# 	tQ=np.zeros(self.num_node)
	# 	for user in self.nodes:
	# 		tQ[user] = self.sample_event( self.mu[user] , 0 , user, T ) 
	# 		# tQ[user] = rnd.uniform(0,T)
	# 	Q=PriorityQueue(tQ)
	# 	# print "----------------------------------------"
	# 	# print "sample event starts"
	# 	t_new = 0
	# 	if flag_check_num_msg==True:
	# 		num_msg = 0
	# 	while t_new < T:

	# 		t_new,u=Q.extract_prior()# do not we need o put back t_new,u 
	# 		u = int(u)

	# 		# print " extracted user " + str(u) + "---------------time : " + str(t_new)
	# 		# t_old=opn_update_time[u]
	# 		# x_old=opn_update_val[u]
	# 		[t_old,x_old] = opn_update[u,:]
	# 		x_new=self.alpha[u]+(x_old-self.alpha[u])*np.exp(-self.w*(t_new-t_old))
	# 		# opn_update_time[u]=t_new
	# 		# opn_update_val[u]=x_new
	# 		opn_update[u,:]=np.array([t_new,x_new])
	# 		m = rnd.normal(x_new,self.var)
	# 		msg_set.append(np.array([u,t_new,m]))
	# 		if flag_check_num_msg == True:
	# 			num_msg = num_msg + 1 
	# 			if num_msg > max_msg :
	# 				break
	# 		# update neighbours
	# 		for nbr in np.nonzero(self.edges[u,:])[0]:
	# 			# print " ------------for nbr " + str(nbr) + "-------------------------"
	# 			# change above 
	# 			[t_old,lda_old] = int_update[nbr,:]
	# 			lda_new = self.mu[nbr]+(lda_old-self.mu[nbr])*np.exp(-self.v*(t_new-t_old))+self.B[u,nbr]# use sparse matrix
	# 			int_update[nbr,:]=np.array([t_new,lda_new])
	# 			t_old,x_old=opn_update[nbr,:]
	# 			x_new=self.alpha[nbr]+(x_old-self.alpha[nbr])*np.exp(-self.w*(t_new-t_old))+self.A[u,nbr]*m
	# 			opn_update[nbr,:]=np.array([t_new,x_new])

	# 			# print " updated int " + str(lda_new) + " ------------ updated opinion -----" + str(x_new)
	# 			t_nbr=self.sample_event(lda_new,t_new,nbr, T )
	# 			# print " update next event time of " + str( nbr ) + "  as " + str(t_nbr)

	# 			Q.update_key(t_nbr,nbr) 
	# 		# Q.print_elements()
			
	# 	if return_only_opinion_updates == True:
	# 		return opn_update
	# 	else:
	# 		return opn_update, int_update, np.array(msg_set) 
	# def sample_event(self,lda_init,t_init,v, T ): # to be checked
	# 	lda_upper=lda_init
	# 	t_new = t_init
		 
		
	# 	# print "------------------------"
	# 	# print "start tm "+str(t_init) + " --- int --- " + str(lda_init)
	# 	# print "------------start--------"
	# 	while t_new < T : #* get T 
	# 		u=rnd.uniform(0,1)
	# 		t_new = t_new - math.log(u)/lda_upper
	# 		# print "new time ------ " + str(t_new)
	# 		lda_new = self.mu[v] + (lda_init-self.mu[v])*np.exp(-self.v*(t_new-t_init))
	# 		# print "new int  ------- " + str(lda_new)
	# 		d = rnd.uniform(0.9,1)
	# 		# print "d*upper_lda : " + str(d*lda_upper)
	# 		if d*lda_upper < lda_new  :
	# 			break
	# 		else:
	# 			lda_upper = lda_new
	# 	return t_new
	# def set_msg(self, flag_only_train = True, msg_set = [], train_fraction = 0):
	# 	if flag_only_train == True:
	# 		self.train = msg_set
	# 		self.num_train = self.train.shape[0]
	# 	else:
	# 		num_msg = msg_set.shape[0]	
	# 		num_tr = int( num_msg * train_fraction )
	# 		self.train = msg_set[:num_tr, :]
	# 		self.test = msg_set[num_tr : ,:]
	# 		self.num_train = self.train.shape[0]
	# 		self.num_test = self.test.shape[0]
	# 		del msg_set
	# def check_user_start_index(self):
	# 	# nodes in msg
	# 	nodes = np.unique(  self.train[:,0])
	# 	node_min = np.amin(nodes)
	# 	node_max = np.amax( nodes)
	# 	node_range_msg = node_max - node_min + 1 
	# 	if node_range_msg == self.num_node:
	# 		print "end nodes appear in msg"
	# 		if node_min == 1:
	# 			print " min node is 1 , therefore shifting users"
	# 			return True
	# 		else:
	# 			return False 
	# 	else:
	# 		print "Not all nodes appeared in msg"
	# 		if node_min == 1:
	# 			print "node min is 1 "
	# 			print "please check further on this dataset"
	# 			return True
	# 	return False
	# print nodes.shape[0]
	# print "-----"
	# print self.num_node
	# print "-----"
	# print np.amax( nodes)
	# print "  -- min -- "
	# print np.amin(nodes)



	#-----------------------------------------------------------------------------


	#--------------- Testing ------------------------------------------------------



	
	#******************** test *******************************************
	# print type(rnd.uniform(size = num_nbr ))
	# def inverse_mat(self,mat):
	# 	return LA.inv(mat)
	# def theta(m):
	# 	pass
	# def parameters(A,tol):
	# 	if cond ==True:
	# 		pass
	# def exponent_mat(self,A,B,t):
	# 	# balance # if self.check_balance()==True:

	# 	mu=np.trace(A)/nuser # define
	# 	A-= mu*np.eye(nuser)
	# 	if t*LA.norm(A,1) == 0 :
	# 		m_star,s = 0,1
	# 	else:
	# 		m_star,s=parameters(t*A,tol)# define 
	# 	F=B
	# 	eta=np.exp(t*mu/s)
	# 	for i in range(s):# define 
	# 		c1=LA.norm(B, np.inf)
	# 		for j in range(m_star):# define
	# 			B=(t/(s*j))*A.dot(B)
	# 			c2=LA.norm(B, np.inf)
	# 			F=F+B
	# 			if c1+c2 <=tol*LA.norm(F, np.inf):
	# 				break
	# 			c1=c2
	# 		F=eta*F
	# 		B=F
	# 	if self.check_balance()==True:
	# 		# define
	# 		return D.dot(F)
	# 	return F
