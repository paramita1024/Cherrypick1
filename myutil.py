# from slant import results
import matplotlib.pyplot as plt
import pickle
import numpy as np
def get_FR(s,t):
	# print float(s.shape[0] - np.count_nonzero(np.sign(s) + np.sign(t))) / s.shape[0]
	return float(s.shape[0] - np.count_nonzero( np.sign(s) + np.sign(t)))/s.shape[0]
def get_MSE(s,t):
	return np.mean((s-t)**2)
def save(obj,output_file):
	with open(output_file+'.pkl' , 'wb') as f:
		pickle.dump( obj , f , pickle.HIGHEST_PROTOCOL)

def load_data(input_file):
	with open(input_file+'.pkl','rb') as f:
		data = pickle.load(f)
	return data
def find_num_msg_plot( msg, num_node, mu):
	num_msg_per_user = np.zeros(num_node)
	for user in range(num_node):
		num_msg_per_user[user] = np.count_nonzero(msg[:,0]==user)
	plt.plot( num_msg_per_user, 'r')
	# plt.plot( self.curr_int, 'b')

	plt.plot( mu, 'b')
	plt.show()

# # Name as myutil
# class myutil:
# 	def __init__(self):
# 		pass
# 	def get_FR(self,s,t):
# 		# print float(s.shape[0] - np.count_nonzero(np.sign(s) + np.sign(t))) / s.shape[0]
# 		return float(s.shape[0] - np.count_nonzero( np.sign(s) + np.sign(t)))/s.shape[0]
# 	def get_MSE(self,s,t):
# 		return np.mean((s-t)**2)

	# def get_initial_opn_int( self, train, end_time ):
	# 	num_train_known = np.count_nonzero(train[:,1] < end_time)
		

	


	# def find_opn_markov( self, check_the_arguements ): # or perhaps next one 
	# 	pass

	# init_opn, init_intensity = myutil.get_initial_opn( self.train, end_time )
	# msg_set, last_opn_update, last_int_update =  myutil.simulate_events(time_span_input , init_opn, init_intensity, self.A, self.B)
	# prediction_array[simulation_no] = myutil.predict_from_events( last_opn_update,  user ) # or perhaps next one 
	# prediction_array[simulation_no] = myutil.find_opn_markov( check_the_arguements ) # or perhaps next one 
	# prediction_array[simulation_no] = myutil.predict_from_events( self.alpha[user], self.A[:,user] , last_opn_update,  msg_set, user , time )
