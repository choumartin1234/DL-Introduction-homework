# -*- encoding: utf-8 -*-

import numpy as np

class MaxPoolingLayer():
	def __init__(self, kernel_size, pad):
		'''
		This class performs max pooling operation on the input.
		Args:
			kernel_size: The height/width of the pooling kernel.
			pad: The width of the pad zone.
		'''

		self.kernel_size = kernel_size
		self.pad = pad
		self.trainable = False

	def forward(self, Input):
		'''
		This method performs max pooling operation on the input.
		Args:
			Input: The input need to be pooled.
		Return:
			The tensor after being pooled.
		'''
		############################################################################
		self.Input = Input
		input_after_pad = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
		n, c_in, h_pad, w_pad = input_after_pad.shape
		h_out = h_pad // self.kernel_size
		w_out = w_pad // self.kernel_size
		
		tmp = np.full((n, c_in, h_out, w_pad), -1000000)
		for i in range(self.kernel_size):
			tmp = np.maximum(tmp, input_after_pad[:, :, i:h_pad:self.kernel_size, :])
			
		output = np.full((n, c_in, h_out, w_out), -1000000)
		for i in range(self.kernel_size):
			output = np.maximum(output, tmp[:, :, :, i:w_pad:self.kernel_size])
		
		self.saved = input_after_pad >= np.kron(output, np.ones((self.kernel_size, self.kernel_size))) 

		return output
	    ############################################################################

	def backward(self, delta):
		'''
		Args:
			delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
		Return:
			delta of previous layer
		'''
		############################################################################
		return self.saved * np.kron(delta, np.ones((self.kernel_size, self.kernel_size)))
	    ############################################################################
