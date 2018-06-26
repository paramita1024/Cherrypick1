import os
from myutil import *
import time
import numpy.ma as ma
# from test_function_call import *
# from spg import *
# import bamos_opt
from spg_new import spg
# import torch
import matplotlib.pyplot as plt
# from ctypes import *
import numpy as np 
import numpy.random as rnd
import pickle
from math import floor
import sys
from numpy import linalg as LA
import math

class results : 
	def __init__(self, result_val_MSE, result_val_FR, predicted_val, original_val):
		self.result_val_MSE = result_val_MSE 
		self.result_val_FR = result_val_FR 
		self.predicted_val = predicted_val
		self.original_val = original_val


class test:
	def __init__(self,a,b,c):
		self.a = a 
		self.b = b
		self.c = c
	def print_it(self):
		print "check"
		print "a"
		print self.a
		print "b"
		print self.b
		print "c" 
		print self.c 
# def test_it( num_var ):
# 	s = np.array([3 , 4, 5])
# 	t = np.array([6,7,8])
# 	u = np.array([1,2,3])
# 	if num_var == 1:
# 		return s 
# 	if num_var == 2:
# 		return s, t 
# 	if num_var == 3 : 
# 		return s,t,u
def test_least_square():
	d0 = 30
	d1 = 10
	var = .00001 
	A = rnd.rand( d0, d1)
	x = rnd.rand( d1)
	# print 
	error = rnd.normal( 0 , var , d0 )
	# print error.shape
	# error = error.reshape(error.shape[0],1)
	# print error.shape
	# # print type(error)
	b = A.dot(x) + error
	print b.shape
	# error_loss = []
	# error_in_x = []
	# for self.lambda_least_square in [ 10^x for x in range(-5,5,1) ]:
	# 	# x_computed = self.solve_least_square(A,b)
	# 	x_computed = LA.lstsq(A,b)[0]
	# 	print x_computed.shape
	# 	# print x_computed
	# 	# error_loss.append( LA.norm(A.dot(x_computed) - b ))
	# 	error_in_x.append(LA.norm( x - x_computed ))

	# 	# plt.plot(error_loss)
	# 	# # plt.ylabel('some numbers')
	# 	# plt.show()
	# # print "error in Ax-b"
	# # print error_loss
	# print "error in x - x*"
	# print error_in_x
def generate_function( d0, d1):
	A = rnd.rand( d0, d1 )
	b = rnd.rand( d1 )
	return A/A.trace(),b/LA.norm(b)

def spectral_proj_grad( A, b ):
	spectral_nu = 10^(-3)
	max_itr = 1000
	likelihood_val = np.zeros( max_itr )
	grad_val = np.zeros( max_itr )
	# mu=np.ones(self.num_node)
	# B=np.zeros((self.num_node,self.num_node))

	# #  gradient calculation
	# coef_mat = np.zeros((self.num_train, self.num_node))
	# last_coef_val = np.zeros( self.num_node)
	# num_msg_user = np.zeros(self.num_node)
	# msg_time_exp = np.zeros(self.num_train)
	# msg_index = 0 
	# for user , time , sentiment  in self.train:
	# 	user = int(user)
	# 	neighbours_with_user  = np.concatenate(([user], np.nonzero(self.edges)[0] ))
	# 	value = np.exp(-self.v * time)
	# 	msg_time_exp[msg_index] = 1/value
	# 	print user
	# 	last_coef_val[user] = last_coef_val[user] + value
	# 	coef_mat[msg_index,neighbours_with_user] = last_coef_val[neighbours_with_user]
	# 	num_msg_user[user] = num_msg_user[user] + 1
	# 	msg_index = msg_index + 1 
	
	# for user in self.nodes:


	# self.mu[user], self.B[user,] =self.spectral_proj_grad()
	# B_user_init = np.ones( self.num_node ) # change
	# x=np.concatenate(([mu[user]], B_user_init))
	# func_val_list=[]
	dim = A.shape[1]
	x = np.ones( dim )
	d = np.ones( x.shape[0] ) 
	H = np.eye( x.shape[0] ) 
	alpha_bb = min([.0001 +.5*rnd.uniform(0,1) , 1 ])
	alpha_min = .0001
	# compute parameters***************  
	# user_msg_index = np.where( self.train[:,0]==user )[0]
	# coef_mat_user =  coef_mat[user_msg_index,:]
	# msg_time_exp_user = msg_time_exp[user_msg_index]
	# user_mask = self.edges[user,:]
	# user_mask[user] = 1

	num_itr = 0 
	
	while LA.norm(d) > sys.float_info.epsilon:

		grad_f , likelihood_val[num_itr] = grad_f_n_f(A,b,x)
		grad_val[num_itr] = LA.norm( grad_f )
		# print grad_f 

		# print "--------------------function value-----------"

		# print likelihood_val
		alpha_bar=min([alpha_min,max(alpha_bb, alpha_min)]) 
	# 	# grad_f = np.ones(self.num_node+1) #**
	# # 	# alpha
		d=project_positive(x-alpha_bar*grad_f)-x #
	# 	# func_val_list.append(self.f(x)) #
	# 	# if len(func_val_list) > self.size_of_function_val_list: 
	# 		# func_val_list.pop(0)
	# 	# max_func_val=max(func_val_list) 
		alpha = 1


		while ( alpha*d.dot(grad_f) + np.power(alpha,2)*(d.dot(H.dot(d))) > spectral_nu*alpha*(grad_f.dot(d)) ) & (alpha > 0) : # lhs # write B as H 
			alpha = alpha - .1
			print "alpha is " + str(alpha)
		# print alpha
		
		s = alpha * d
		x = x + s


		y = H.dot(d)
		alpha_bb = y.dot(y) / s.dot(y) 
		# # Bk=Bk-Bk*(sk*sk')*Bk/(sk'*Bk*sk)+yk*yk'/(yk'*sk);
		H_s = H.dot(s).reshape(H.dot(s).shape[0],1)
		y_2dim = y.reshape(y.shape[0],1)
		H = H - s.dot(H.dot(s)) * np.matmul( H_s , H_s.T ) + np.matmul( y_2dim , y_2dim.T )/ y.dot(s)
		
		num_itr = num_itr + 1
		if num_itr == max_itr:
			break # to be deleted
			
	plt.plot(likelihood_val)
	plt.show()
	plt.plot(grad_val)
	plt.show()
	# self.mu = mu
	# self.B = B


def test_least_square():
	d0 = 300
	d1 = 100
	var = .01 
	A = rnd.rand( d0, d1)
	x = rnd.rand( d1)
	# print x.shape
	error = rnd.normal( 0 , var , d0 )
	# print error
	# error = error.reshape(error.shape[0],1)
	# print error.shape
	# print type(error)
	b = A.dot(x) + error
	# print A.dot(x).shape
	error_loss = []
	error_in_x = []
	x_axis = np.arange(d1)
	# print x_axis.shape
	for lda in [1]:#[ 10^x for x in range(-5,5,1) ]:
		x_computed = solve_least_square(A,b,lda)
		# x_computed = LA.lstsq(A,b)
		# print x_computed.shape
		print x_computed
		# error_loss.append( LA.norm(A.dot(x_computed) - b ))
		print LA.norm( x - x_computed )
		# error_in_x.append(LA.norm( x - x_computed ))
		# plt.plot(error_in_x)
		# print x_axis.shape
		# print x.shape
		# print x_computed.shape
		plt.plot(x_axis, x, 'r--', x_axis, x_computed, 'bs')
		plt.show()
		

	# 	# plt.plot(error_loss)
	# 	# # plt.ylabel('some numbers')
	# 	# plt.show()
	# # print "error in Ax-b"
	# # print error_loss
	# # print "error in x - x*"
	# # print error_in_x
	# plt.plot(error_in_x)
	# plt.show()
	


def project_positive(v):
	return np.maximum(v,np.zeros(v.shape[0]))
# def grad_f(A,b,x):
# 	tmp = np.reciprocal(  A.dot(x) )
# 	grad_f_val = b - A.T.dot(tmp) 
# 	return grad_f_val
# def f(A,b,x):
# 	f_val =  x.dot(b) - np.sum( np.log( A.dot(x) ) ) 
# 	return f_val


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
		tmp = self.coef_mat_user.dot(b) * self.msg_time_exp_user + mu
		# print tmp
		t1 = np.sum( np.log( self.coef_mat_user.dot(b) * self.msg_time_exp_user + mu ) )
		# print t1

		t2 = mu*self.last_time_train - (  np.exp(-self.v*self.last_time_train)*(b.dot(self.last_coef_val_user)) - b.dot(self.num_msg_user)) / self.v	
		# print t2
		return ( t2 - t1 )
		
	def grad_f(self,x):
		mu = x[0]
		b = x[1:]
		tmp = self.coef_mat_user.dot(b) * self.msg_time_exp_user + mu

		common_term = np.reciprocal(self.coef_mat_user.dot(b) * self.msg_time_exp_user + mu)
		# print common_term 
		del_b_t1 = self.coef_mat_user.T.dot( common_term *  self.msg_time_exp_user)
		del_mu_t1= np.sum(common_term)
		# print del_mu_t1


		del_b_t2 = -(np.exp(-self.v*self.last_time_train)*(self.last_coef_val_user) - self.num_msg_user) / self.v
		del_mu_t2= self.last_time_train
		# tmp = np.exp(-self.v*self.last_time_train)*(self.last_coef_val_user) 
		# print tmp
		# print del_b_t2
		del_t1 = np.concatenate( ([del_mu_t1], del_b_t1))
		del_t2 = np.concatenate( ([del_mu_t2], del_b_t2))
		grad_f = del_t2 - del_t1 
		print "done-------------------------"
		return grad_f 
	
# class test_me:
# 	def __init__(self,d):
# 		self.d = d 
# 	def func(self,d2):
# 		return self.d+d2
# def test_it(g,x):
	
# 	return  g(x)

def inner(n):
	def inner_inner(n1):
		return n1+n 
	return n+inner_inner(n)
def sample_event(mu, lda_init,t_init,v, T ): # to be checked
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
			lda_new = mu + (lda_init-mu)*np.exp(-v*(t_new-t_init))
			# print "new int  ------- " + str(lda_new)
			d = rnd.uniform(0,1)
			# print "d*upper_lda : " + str(d*lda_upper)
			if d*lda_old < lda_new  :
				break
			else:
				lda_old = lda_new
			# itr += 1
				

		return t_new # T also could have been returned
def predict_from_events( alpha, A , last_opn_update,  msg_set, user , time ): # confirm from abir da whether to send alpha or curr opn variable of self 
	w=4
	time_array = msg_set[:,1]
	time_last, opn  = last_opn_update 
	# print msg_set[np.logical_and((time_array > time_last ), ( time_array < time)) ,:]
	for user_curr, time_curr, sentiment in msg_set[np.logical_and((time_array > time_last ), ( time_array < time)) ,:]: 
		time_diff = time_curr - time_last
		opn = alpha + (opn - alpha )*np.exp(-w*time_diff)+A[int(user_curr)]*sentiment
		print opn
		time_last = time_curr
	return opn
def solve_least_square(A,b):
	A_T_b = A.T.dot(b) # check 
	mat = np.matmul(A.T,A) + (np.eye( A_T_b.shape[0] )) 
	# print "rank" + str(LA.matrix_rank(mat))
	# print mat
	# print "size" + str(mat.shape[0]) + "," + str(mat.shape[1])
	# for i in range(12):
	# 	if LA.norm(mat[:,i]) == 0 : 
	# 		print "col zero"
	x=LA.solve( mat , A_T_b)
	return x[0],x[1:]
def find_alpha_A():
	train = np.array([[0,1,-1],[1,1.5,1],[2,2,-1],[0,3,1],[1,4,-1],[2,5,1]])
	w=4
	nodes = np.arange(3)
	num_node = 3
	edges = np.array([[0,1,0],[1,0,1],[0,1,0]]) 
	alpha = np.zeros( num_node)
	A = np.zeros(( num_node,  num_node ))
	index={}
	for user in  nodes : 
		index[user] = np.where( train[:,0]==user)[0]
	# print index 
	for user in  nodes :
		# print "-------------------------------------------------------"
		# print user
		# print "--------------------------------------------------------"
		num_msg_user = index[user].shape[0]
		if num_msg_user > 0 :
			neighbours = np.nonzero( edges[user,:])[0]
			num_nbr = neighbours.shape[0]
			g_user = np.zeros( ( num_msg_user, num_nbr ) )
			msg_user =  train[index[user],2]
			nbr_no=0
			for nbr in neighbours :
				# print "nbr -------------------" + str(nbr)
				if index[nbr].shape[0] > 0 :
					user_msg_ind = 0
					time = 0 
					opn = 0 
					index_for_both = np.sort( np.concatenate((index[user], index[nbr])) )
					for ind in index_for_both :
						# print ind 
						user_curr , time_curr , sentiment =  train[ind,:]
						if user_curr == user:
							opn = opn*np.exp(- w*(time_curr - time))
							g_user[user_msg_ind, nbr_no]=opn
							user_msg_ind = user_msg_ind + 1
							if user_msg_ind == num_msg_user:
								break
						else:
							opn = opn*np.exp(- w*(time_curr - time))+sentiment
						time = time_curr
				nbr_no = nbr_no + 1
			g_user = np.concatenate(( np.ones((msg_user.shape[0],1)) , g_user  ), axis = 1 )
			# print g_user
			alpha[user], A[neighbours,user] =  solve_least_square( g_user, msg_user )
	print "alpha -------------------------------"
	print alpha
	print "A-------------------------------------"
	print A
	 # alpha = alpha
	 # A = A
def find_mu_B():
	#-------------------------
	# init_function_val = np.zeros( num_node)
	# end_function_val =  np.zeros( num_node)
	#-------------------------
	num_train = 8
	train = np.array([[0,1,-1],[1,2,1],[2,3,-1],[3,4,1],[0,5,-1],[1,6,1],[2,7,-1],[3,8,1]])
	w=4
	v=2
	nodes = np.arange(4)
	num_node = 4
	edges = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,0],[1,0,0,0]]) 
	
	mu=np.zeros( num_node)
	B=np.zeros(( num_node, num_node))
	spg_obj = spg() 
	#  gradient calculation
	coef_mat = np.zeros(( num_train,  num_node))
	last_coef_val = np.zeros(  num_node)
	num_msg_users = np.zeros( num_node)
	msg_time_exp = np.zeros( num_train)
	msg_index = 0 
	for user , time , sentiment  in  train:
		user = int(user)
		neighbours_with_user  = np.concatenate(([user], np.nonzero( edges[user,:])[0] ))
		value = np.exp( v * time)
		# print value
		msg_time_exp[msg_index] = 1/value
		# print user
		last_coef_val[user] +=  value
		coef_mat[msg_index,neighbours_with_user] = last_coef_val[neighbours_with_user]
		num_msg_users[user] = num_msg_users[user] + 1

		msg_index = msg_index + 1 
	# print coef_mat
	# print msg_time_exp
	# print num_msg_users
	for user in  [2] : #nodes:
		#  mu[user],  B[user,] = spectral_proj_grad()
		neighbours_with_user  = np.concatenate(([user], np.nonzero( edges[user,:])[0] ))
		# print neighbours_with_user
		x0  = np.ones(1+ neighbours_with_user.shape[0])
		# print x0
		# compute parameters************
		user_msg_index = np.where(  train[:,0]==user )[0]
		# print user_msg_index
		# print "-------check-----------"
		# print user_msg_index
		# print "-----------------------"
		# print neighbours_with_user

		coef_mat_user =  coef_mat[user_msg_index.reshape( user_msg_index.shape[0],1),neighbours_with_user.reshape(1, neighbours_with_user.shape[0])]
		# print "All"
		# print coef_mat
		# print "2"
		# print coef_mat_user
		# print msg_time_exp
		msg_time_exp_user = msg_time_exp[user_msg_index]
		# print msg_time_exp_user
		# print last_coef_val
		last_coef_val_user = last_coef_val[neighbours_with_user]
		# print last_coef_val_user
		num_msg_user=num_msg_users[neighbours_with_user]
		last_time_train =  train[-1,1]
		# print last_time_train
		#-------------------------------------------
		grad_function_obj = Grad_n_f(coef_mat_user, last_coef_val_user, num_msg_user, msg_time_exp_user, last_time_train,  v)
		print grad_function_obj.grad_f(np.array([1,2,3]))
		# # init_function_val[user] = grad_function_obj.f(x0)
		# result= spg_obj.solve( x0, grad_function_obj.f, grad_function_obj.grad_f,  project_positive,  queue_length , sys.float_info.epsilon,  max_iteration_mu_B )
		# x = result['bestX']
		# # end_function_val[user] = grad_function_obj.f(x)
		# mu[user] = x[0]
		# B[neighbours_with_user, user] = x[1:]	

def load_results(result_file):
	result_obj = load_data(result_file)
	print "MSE  " + str(result_obj.result_val_MSE)
	print "FR  " + str(result_obj.result_val_FR)
	plt.plot(result_obj.predicted_val, "r")
	plt.plot(result_obj.original_val, "b")
	plt.show()
def main():

	# result_val_MSE, result_val_FR, predicted_val, original_val

	r_obj = results(1,2,np.array([1,2,3]), np.array([3,4,5]))
	save( r_obj, "result_file")
	del r_obj
	load_results("result_file")
	# print np.version.version
	# print np.log
	# print np.exp()
	# print 13-.1624
	# print 3+3*np.exp(-2)+3.4086-12.8376#+2*np.exp(-8)+3*np.exp(-10)  #+np.exp(-4)
	# print "------------------------------"
	# find_mu_B()
	# print 1/np.exp(8)
	# print np.arange(4)*np.arange(4)+3
	# A = np.arange(4).reshape(2,2)
	# b = np.arange(5)
	# A[1,[2,3,4]]= b[[2,3,4]]
	# print A
	
	# # print A
	# b = np.array([1,2])
	# print np.matmul(A.T,A)+ np.eye(2)
	# find_alpha_A()
	# print np.sort(np.array([4,2,3,1]))
	# a = np.arange(10)#.reshape(2,5)
	# print np.where(a>5)[0]
	# print np.nonzero(a>5)[0]
	# a = 3
	# if (a > 7 ) or (a < 3):
	# # 	print a
	# print -1*np.exp(-2)
	# print -np.exp(-12)+np.exp(-4)
	# print 0
	# print -np.exp(-8)
	
	# msg_set = np.array([[0,1,-1],[1,2,1],[2,3,.1],[1,5,.6]])
	# alpha = 0
	# A = np.array([0,.1,.1])
	# last_opn_update = np.array([1,-.9])
	# user = 0
	# time = 7
	# predict_from_events( alpha, A , last_opn_update,  msg_set, user , time )
	# a = np.arange(16).reshape(4,4)
	# print a> 1
	# print a[[2,3],:]
	# tQ=rnd.rand(4)
	# print tQ
	# for i in range(4):
	# 	u = np.argmin(tQ)

	# 	t_new = tQ[u]
	# 	tQ[u] = float('inf')

	# 	print tQ
	# print np.concatenate((np.array([1,2,3,4]).reshape(4,1), np.array([5,6,7,8]).reshape(4 , 1 )), axis=1)
	# mu = rnd.rand(1)*10
	# tlist= []
	# for i in range(100):
	# 	tlist.append( sample_event(mu= mu, lda_init= 10,t_init=3 ,v=2, T=4 ))
	# plt.plot( tlist,'r')
	# tlist1= []
	# for i in range(100):
	# 	tlist1.append( sample_event(mu= mu, lda_init= 20,t_init=3 ,v=2, T=4 ))
	# plt.plot( tlist1,'b')
	# plt.show()

	# print range(0,5,1)
	# filename = "current_status"
	# if os.path.exists(filename):
	# 	os.remove(filename)

	# for i in range(20):
	# 	time.sleep(5)
	# 	with open( filename ,'a+') as f :
	# 		f.write(str(i) + "th msg complete\n")
	# a = rnd.rand(10)
	# b= rnd.rand(10)

	# print get_FR(a,b)
	# print get_MSE(a,b)
	# print math.log(np.exp(-1))
	# a=np.arang1rgmax(a)
	# print inner(3)
	# n = 1000
	# a = rnd.random(n)
	# b = list(a)
	# st = time.time()
	# c = np.mean(a)
	# print time.time() - st

	# st = time.time()
	# d = sum(b) / float(len(b))
	# print time.time() - st	
	# # print a
	# # call eig
	# st = time.time()
	# b = np.sum(np.reciprocal(LA.eigvals(a)))
	# et = time.time()
	# print et-st
	# # call inv 

	# st = time.time()
	# a_inv = np.linalg.inv(a)

	# # print a_inv
	# b = np.sum(np.diag(a_inv))
	# et = time.time()
	# print et-st
	# print b
	#-----------------------------------------
	# a = np.0
	# print np.diag(a, k = 1)
	# a = ma.make_mask([True, False, True, False])
	# # print type(a)
	# print a.nonzero()
	# Y = np.arange(16).reshape(4,4)
	# Y[1] = np.array([11,12,13,14])
	# print Y[:,1].shape
	# print a
	# print b 
	# print c 
	# print d 
	# print Y[np.array([0,3]),:][:,[1,2]]
	# print Y[np.array([0,3]).reshape(2,1) , np.array([1,2]).reshape(1,2)]
	# a = np.zeros((3,4))
	# b = np.array([0,2])
	# c = np.array( [ 0,3])
	# print a[ b , c ]
	# print b.shape
	# print 2*math.exp(40)
	# a = np.array([1,2])
	# print a.shape[0]
	# 	msg = np.array([[3,1,2],[2,0,1],[4,-1,5],[1,1,1]])
	# 	# print np.argsort(msg[:,1])
	# 	msg = msg[np.argsort( msg[:,1] ), :]
	# 	# print msg
	# print "do nothing"
	# t = test_me(1)

	# test_it(t.func)
	# spg_obj = spg()
	# d0 = 4
	# d1 = 2
	# x0 = rnd.rand( d1)
	# A,b = generate_function(d0, d1)
	# grad_function_obj = Grad_n_f(A,b)
	# # value = test_it( g_obj.grad_f, x )
	# # print value

	# max_iteration = 100
	# queue_length = 100
	# result= spg_obj.solve( x0, grad_function_obj.f, grad_function_obj.grad_f, project_positive, queue_length , sys.float_info.epsilon, max_iteration )
	# print result['buffer']
	# plt.plot(result['buffer'])
	# plt.show()
	# print result['bestX']
	# def solve(x0, f, g, proj, m, eps, maxit, callback=None):
	# test_function_call(5)
	# print "do nothing"
	# print math.log(.66)
	# test_least_square()
	# A=np.array([[1,2],[3,4],[5,6]])
	# print np.flip(A,)
	# B=A[[0,1],:]
	# A[0,1]=9
	# # print B
	# d0 = 4
	# d1 = 2
	# # # # generate_function(d0, d1)
	# A,b = generate_function(d0, d1)

	# spectral_proj_grad(A, b )
	# A = rnd.rand(10,10)
	# A1 = A[:5,:]
	# A2 = A[5: , :]
	# del A
	# print A1
	# print A2
	# print np.linalg.norm(np.array([1,2,3]))
	# print np.arange(10)
	# test_least_square()
	# # A = np.array([[1,2,3],[4,5,6]])
	# A = np.random.rand(3,3)
	# b=np.array([1,2,3])
	# e= np.random.normal(size=3)
	# print e

	# alpha = np.array([1,2,3])
	# mu = np.array([4,5,6])
	# time_init = np.zeros((3 ,1))

	# opn_update = np.concatenate((time_init, alpha.reshape( 3 , 1 )), axis=1)
	# int_update =  np.concatenate((time_init, mu.reshape( 3 , 1 )), axis=1)
	# # print ctypes.addressof(opn_update)
	# print "---------------opn----------------"
	# print opn_update
	# print "---------------int----------------"
	# print int_update

	# time_init[0] = 35
	# print "---------------opn-----------------"

	# print opn_update
	# print "---------------int-----------------"
	# print int_update

	# print opn_update 

	# print int_update
	# num_var = 3 
	# u = test_it(num_var)
	# print u
	# arr = np.array([1,2,3])
	# print arr
	# test_it(arr)
	# print arr
	# print np.matmul(arr.reshape(arr.shape[0],1) , arr.reshape(1,arr.shape[0]))
	# arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
	# arr_part = arr[[0,2],:]
	# arr[0,0] = 35
	# print arr
	# print arr_part
	# arr = np.array([1,2,3])
	# arr1 = np.array([3,4,5])
	# print np.multiply(arr, arr1)
	# print arr*arr1

	# arr = np.array([1,2,3])
	# print np.concatenate(([4],arr))
	# arr = np.array([[1,2,3],[4,5,6]])
	# for a,b,c in arr:
	# 	print a 
	# 	print b 
	# 	print c
	# v = np.array([1,2,3])
	# print v
	# print v.T
	# print v.shape

	# v1=np.array([v])
	# print v1
	# print v1.T
	# print v1.shape
	# v1 = np.array([[1,0], [0,1]])
	# v2 = np.array([[0],[1]])
	# # v3 = v2.T 
	# v3 = v1.dot(v2)
	# print v3.shape
	# print np.matmul(v1,v2)
	# g=v2.shape
	# print type(g)
	# v = np.array([1,-1,2])
	# v1 = project_positive_quadrant(v)
	# print v1
	# print v
	# t=test(1,2,3)
	# with open('test_file'+'.pkl','wb') as f:
	# 	pickle.dump(t, f, pickle.HIGHEST_PROTOCOL)
	# del t
	# try:
	# 	print t
	# except NameError:
	# 	print "t does not exist"
	# else:
	# 	print " t exists"
	# with open('test_file'+'.pkl','rb') as f:
	# 	t_new = pickle.load(f)
	# t_new.print_it()
if __name__=="__main__":
	main()







# class tree_op:
# 	def __init__(self):
# 		return
# 	def parent(self,i):
# 		if i==0:
# 			print('given index is root')
# 			return -1
# 		return int(floor((i-1)/2))
# 	def left(self,i):
# 		return 2*i+1
# 	def right(self,i):
# 		return 2*i+2


# class Qarray:
# 	def __init__(self,Q):
# 		n=len(Q)
# 		self.Q=np.zeros((n,2))		
# 		for u in range(n):
# 			self.Q[u]=[Q[u],u]
# 	def cmp(self,i,j):
# 		#print type(i)
# 		#print type(j)
# 		return (self.Q[i][0] < self.Q[j][0])
# 	def cmp_with_val(self,key,i):
# 		return ( key < self.Q[i][0] )
# 	def exchange(self,i,j):
# 		temp = np.array(self.Q[i])
# 		self.Q[i]=self.Q[j]
# 		self.Q[j]=temp
# 	def get(self,i):
# 		return self.Q[i]
# 	def update_key(self,i,key):
# 		self.Q[i][0]=key
# 	def check_eq(self,i,j):
# 		return self.Q[i][0]==self.Q[j][0]
# 	def size(self):
# 	 	return self.Q.shape
# 	def get_index_and_val(self,u,stop_index):
# 		for i in range(stop_index):
# 			if self.Q[i][1]==u:
# 				return i,self.Q[i][0]
# 		print "user not found in heap"
# 		return -1,0
# 	def set(self,i,v): 
# 		self.Q[i]=v
		
# class PriorityQueue(tree_op):
# 	def __init__(self,Q):
# 		tree_op.__init__(self)
# 		self.Q=Qarray(Q)
# 		self.flag_user=np.ones(len(Q))
# 		self.heapsize=len(Q)
# 		self.max_size = self.heapsize
# 		self.build_heap()
# 	def build_heap(self):
# 		for i in range(int(floor(self.heapsize/2))-1,-1,-1):
# 			self.heapify(i)
# 	def heapify(self,i):
# 		l=self.left(i) 
# 		r=self.right(i)  
# 		selected=i # selected is the index to be heapified next time
# 		if l<self.heapsize:
# 			if self.Q.cmp(l,selected): 
# 				selected=l
# 		if r<self.heapsize:
# 			if self.Q.cmp(r,selected): 
# 				selected=r
# 		if selected != i:
# 			self.Q.exchange(i,selected) 
# 			self.heapify(selected)
# 	def insert(self,t): 
# 		self.heapsize=self.heapsize+1
# 		if self.heapsize <= self.max_size:
# 			self.Q.set(self.heapsize-1,[float('Inf'),t[1]])
# 			self.minheap_dec_key(self.heapsize-1,t[0])
# 		else:
# 			print "maximum size of heap is reached" 
# 	def extract_prior(self):
# 		if self.heapsize < 1 :
# 			print("heap underflow")
# 		val=np.array(self.Q.get(0))
# 		#print "val"

# 		self.Q.exchange(0,self.heapsize-1)
# 		self.heapsize=self.heapsize-1
# 		self.heapify(0)
# 		self.flag_user[int(val[1])]=0
# 		return val
# 	def minheap_inc_key(self,i,key):
# 		if self.Q.cmp_with_val(key,i): 
# 			print("new key is smaller than current key")
# 		self.Q.update_key(i,key)
# 		self.heapify(i) 
# 	def minheap_dec_key(self,i,key):		
# 		if ~self.Q.cmp_with_val(key,i):
# 			print("new key is not smaller than current key")
# 		self.Q.update_key(i,key)
# 		while i>0:
# 			p=self.parent(i)	
# 			if self.Q.cmp(i,p): 
# 				self.Q.exchange(i,p)
# 				i=p
# 			else:
# 				break
# 		return
		
# 	def update_key(self,t,u):
# 		if self.flag_user[u]==0:
# 			self.flag_user[u]=1
# 			self.insert(np.array([t,u]))
# 		else:
# 			ind,old_t=self.Q.get_index_and_val(u,self.heapsize)
# 			if t > old_t:
# 				self.minheap_inc_key(ind,t)
# 			else:
# 				self.minheap_dec_key(ind,t)
# 	def print_heap(self):
# 		i=0
# 		c=1
# 		while i<self.heapsize:
# 			# for j in range(i,i+c):
# 			# 	print self.Q.Q[j][0]
# 			# print "----------"
# 			if i+c<=self.heapsize:
# 				print self.Q.Q[i:i+c,0]
# 			else:
# 				print self.Q.Q[i:self.heapsize,0]
# 			i=i+c
# 			c=c*2

# Q_init=[50,32,2,21,0,4,1,3,5,6,7]
# # Q=Qarray(Q_init)
# #print Q.Q
# # print Q.cmp(1,2)

# Q=PriorityQueue(Q_init)
# print Q.Q.Q#print_heap()
# #print Q.heapsize
# print Q.extract_prior()
# #print Q.heapsize
# #print Q.flag_user
# Q.update_key(40,5)
# print Q.Q.Q
# #print Q.heapsize
# #print Q.flag_user
# #Q.print_heap()
# # print Q.Q.Q
# # print Q.Q.get_index_and_val(3,Q.heapsize)
# # print "over"
# # Q.minheap_inc_key(2,6)
# # print "updated"
# # Q.print_heap()
# # Q.minheap_dec_key(5,-.25)
# # print "updated"
# # Q.print_heap()


# #a=Q.extract_prior()
# #Q.update_key(7,4) 
# #print a
# #print Q.flag_user
# #Q.print_heap()
# # print range(3,-1)
# # Q=[41,32,65]
# # Q_new=Qarray(Q)
# # print Q_new.check_eq(0,1)
# # Q_new.update_key(1,41)
# # print Q_new.check_eq(0,1)
# # print Q_new.Q.size
# # for i in range(len(Q)):
# # print Q_new.Q
# # print Q_new.check_eq(0,1)
# # Q_new.set(1,np.array([41,1]))
# # print Q_new.Q
# # print size(Q_new.Q)
# # Q_new.check_eq(0,1)
# # Q_new.update_key(1,34)
# # print Q_new.Q

# #print type(Q_new.Q)
# #Q_new.exchange(0,2)
# #print type(Q_new.get(1))
# # t=tree_op()
# # print t.right(2)










# graph generation
