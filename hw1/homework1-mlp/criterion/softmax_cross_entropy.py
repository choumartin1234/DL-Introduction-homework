""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')
		self.logit = 0 # logit is used , need to be recorded 
		self.gt = 0 # gt is used, need to be recorded

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 1)
	    """
		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.accu and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.
		self.logit = logit
		self.gt = gt
		N = gt.shape[0]
		exp = np.exp(logit)
		h = exp / exp.sum(axis = 1)[:, np.newaxis]
		self.loss = -1*(gt * np.log(h)).sum()/N
		result = ( (logit.argmax(axis=1)) == (gt.argmax(axis=1)) )
		self.acc = result.sum()/N
	    ############################################################################
		return self.loss


	def backward(self):
		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
		# Note : gradient = 1/N *( h(logit) - gt ) , where is softmax function
		N = self.gt.shape[0]  
		exp = np.exp(self.logit)
		h = exp / exp.sum(axis = 1)[:, np.newaxis]  # softmax function
		return (h-self.gt)/N
	    ############################################################################
