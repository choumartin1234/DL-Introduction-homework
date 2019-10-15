""" Euclidean Loss Layer """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.acc = 0. # typo. 
		self.loss = 0.
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
		self.loss = 0.5*(((logit-gt)*(logit-gt)).sum())/N
		#result = ((logit.argmax(axis=1)) == (gt.argmax(axis=1)))
		#self.acc = result.sum()/N
		self.acc = (logit.argmax(axis=1) == gt.argmax(axis=1)).sum() / N
	    ############################################################################
		return self.loss

	def backward(self):
		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
		N = self.gt.shape[0]        
		return (self.logit-self.gt)/N
	    ############################################################################
