""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
	def __init__(self):
		"""
		Applies the element-wise function: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False
		self.output = 0 # record the output from forward layer

	def forward(self, Input):
		############################################################################
	    # TODO: Put your code here
		# Apply Sigmoid activation function to Input, and return results.
		self.output = 1/(1+np.exp(-1*Input)) # sigmoid(x) = 1/(1+e^-x)         
		return self.output 
	    ############################################################################

	def backward(self, delta):
		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta
		return delta * self.output * (1-self.output)  # f'(z) = f(z)(1-f(z)), for f(z) = sigmoid
	    ############################################################################
