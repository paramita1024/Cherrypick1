from myutil import *
# from slant import results
import numpy as np
import pickle
class data_preprocess:
	def __init__(self,file ):
		self.file = file
		# self.edges = edges
		# self.train = train
		# self.test = test
		# self.nuser = graph.get_nuser() # to be defined
	def read_data(self):
		# read the graph
		# f = open ( 'input.txt' , 'r')
		# l = [[int(num) for num in line.split(',')] for line in f ]
		# print l
		# with open(self.file + '.metadata', 'r') as f:
		# 	for line in f :
		# 		metadata =  [ int(num) for num in line.split(',') ] 
		# 	# metadata is a list of int , 
		# 	# metadata = [ num_node ]
		# 	self.num_node = metadata[0]
		# 	self.nodes = range( self.num_node )
		# print self.num_node
		# print self.nodes

		with open(self.file + '.graph', 'r') as f:
			self.edges = np.array([ [ int(num) for num in line.split(',') ] for line in f ])
		self.num_node = self.edges.shape[0]
		# print self.num_node
		self.nodes = range( self.num_node )
		# create adjacency matrix
		# print self.edges
		# # read the msg 
		with open(self.file + '.msg', 'r') as f:
			self.msg = np.array([ [ float(num) for num in line.split(',') ] for line in f ])
			self.num_msg  = self.msg.shape[0]
			print "num msg = "+ str(self.num_msg)
		self.msg = self.msg[np.argsort( self.msg[:,1] ), :]
                print "number of user = "+str(self.num_node)
                print "max user index= "+str(np.max(self.msg[:,0]))
                print "min user index "+ str(np.min(self.msg[:,0]))
		# msg = msg[np.argsort( msg[:,1] ), :]

		# print self.msg
		
		# create 3 column array 

		# split the data 

		# create test and train 

		# save as python object 
	def split_data(self, train_fraction = 0 ):
		num_tr = int( self.num_msg * train_fraction )
		self.train = self.msg[:num_tr, :]
		self.test = self.msg[num_tr : ,:]
		print "num_train = " + str(self.train.shape[0])
		print "num_test = " + str(self.test.shape[0])
		del self.msg
		# print self.train

		# print "==========================="
		# print self.test



		
def save_data(obj, write_file):
	# data_obj = data(graph , train, test )
	with open(write_file+'.pkl','wb') as f:
		pickle.dump( obj, f, pickle.HIGHEST_PROTOCOL)

	
# done ----------------------------------
# Twitter_10ALLXContainedOpinionX
# VTwitter_10ALLXContainedOpinionX
# trump _ data 
# real vs ju 703
# MsmallTwitter
# M large
# Juv
# jaya verdict

def main():
	path = '../Cherrypick_others/Data_opn_dyn_python/'
	filename = path + 'VTwitter_10ALLXContainedOpinionX'
	read_file = filename + '.mat'
	write_file = filename + '.obj'
	fraction_of_train = .9
	dp = data_preprocess(read_file)
	dp.read_data()
	dp.split_data(fraction_of_train)
	save_data(dp , write_file )
if __name__ == "__main__":
	main()

