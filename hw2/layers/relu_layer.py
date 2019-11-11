""" ReLU Layer """

import numpy as np

class ReLULayer():
	def __init__(self):
		"""
		Applies the rectified linear unit function element-wise: relu(x) = max(x, 0)
		"""
		self.trainable = False # no parameters
		self.input = 0 # record the Input from forward layer

	def forward(self, Input):
		############################################################################
	    # TODO: Put your code here
		# Apply ReLU activation function to Input, and return results.
		self.input = Input
		return np.maximum(Input,0)
	    ############################################################################


	def backward(self, delta):
		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta
		return delta * (self.input > 0)  # f'(x) = 0 , if x <= 0 ; 1, if x >0 for ReLU
	    ############################################################################
