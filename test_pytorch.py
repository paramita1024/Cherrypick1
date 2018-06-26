import torch
from numpy import linalg as LA
from torch.autograd import Variable
import matplotlib.pyplot as plt
# import scipy
def main():
	# print "my name"
	d0 = 2
	d1 = 3
	max_iter = 1000
	A = torch.rand(d0,d1)
	# print(A)
	b = torch.rand(d1)
	# print b
	
	x= Variable(  torch.rand(d1), requires_grad = True )
	# x.requires_grad_(True)
	# print x
	loss = torch.mv(A,x)
	# print loss
	loss = torch.log(loss)
	# print loss

	loss = torch.sum(loss)
	# loss.requires_grad_(True)
	print loss
	print "----------------------------------------"
	
	optimizer = torch.optim.Adam( [x] , lr = .00001)
	loss_val = [] 
	for itr in range(max_iter):
		loss = torch.mv(A,x)
		
		loss = torch.log(loss)
		

		loss = torch.sum(loss)
		print x, loss
		optimizer.zero_grad()
		loss.backward(retain_graph=True) # retain_graph=True
		optimizer.step()
		# print loss
		# x -= lr* loss.grad
		loss_val.append(loss.item())
	plt.plot(loss_val)
	plt.show()

	# pass
if __name__=="__main__":
	main()