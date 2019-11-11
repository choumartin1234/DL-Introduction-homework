# -*- encoding: utf-8 -*-

import numpy as np

# if you implement ConvLayer by convolve function, you will use the following code.
from scipy.signal import fftconvolve as convolve

class ConvLayer():
	"""
	2D convolutional layer.
	This layer creates a convolution kernel that is convolved with the layer
	input to produce a tensor of outputs.
	Arguments:
		inputs: Integer, the channels number of input.
		filters: Integer, the number of filters in the convolution.
		kernel_size: Integer, specifying the height and width of the 2D convolution window (height==width in this case).
		pad: Integer, the size of padding area.
		trainable: Boolean, whether this layer is trainable.
	"""
	def __init__(self, inputs, filters, kernel_size, pad, trainable=True):
		self.inputs = inputs
		self.filters = filters
		self.kernel_size = kernel_size
		self.pad = pad
		assert pad < kernel_size, "pad should be less than kernel_size"
		self.trainable = trainable

		self.XavierInit()

		self.grad_W = np.zeros_like(self.W)
		self.grad_b = np.zeros_like(self.b)

	def XavierInit(self):
		raw_std = (2 / (self.inputs + self.filters))**0.5
		init_std = raw_std * (2**0.5)

		self.W = np.random.normal(0, init_std, (self.filters, self.inputs, self.kernel_size, self.kernel_size))
		self.b = np.random.normal(0, init_std, (self.filters,))

	def forward(self, Input):
		'''
		forward method: perform convolution operation on the input.
		Agrs:
			Input: A batch of images, shape-(batch_size, channels, height, width)
		'''
		############################################################################
		self.Input = Input
		kernel_size = self.kernel_size
		input_after_pad = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
		n, c_in, h_pad, w_pad = input_after_pad.shape
		h_out = h_pad - kernel_size + 1
		w_out = w_pad - kernel_size + 1
		c_out = self.W.shape[0]
		output = np.zeros((n, c_out, h_out, w_out))
		for i in range(c_out):
			for j in range(c_in):
				ker = np.rot90(self.W[i, j], 2)[np.newaxis, :, :]
				output[:, i] += convolve(input_after_pad[:, j], ker, 'valid')
		output += self.b[np.newaxis, :, np.newaxis, np.newaxis]
		return output
	    ############################################################################


	def backward(self, delta):
		'''
		backward method: perform back-propagation operation on weights and biases.
		Args:
			delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
		Return:
			delta of previous layer
		'''
		############################################################################
		Input = self.Input
		kernel_size = self.kernel_size
		n, c, h_out, w_out = delta.shape
		n, c_in, h_in, w_in = Input.shape
		h_pad, w_pad = h_in + 2 * self.pad, w_in + 2 * self.pad
		c_out = self.W.shape[0]
		delta_after_pad = np.zeros((n, c_in, h_pad, w_pad))
		for i in range(c_in):
			for j in range(c_out):
				delta_after_pad[:, i] += convolve(delta[:, j], self.W[j, i][np.newaxis, :, :], 'full')
		delta_out = delta_after_pad[:, :, self.pad:h_in + self.pad, self.pad:w_in + self.pad]   # 去掉pad
		input_after_pad = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
		self.grad_W = np.zeros((c_out, c_in, kernel_size, kernel_size))
		for i in range(c_in):
			for j in range(c_out):
				tmp = np.flip(np.rot90(delta[:, j], 2, (1, 2)), 0)
				self.grad_W[j, i] += convolve(input_after_pad[:, i], tmp, 'valid').squeeze()
		self.grad_b = np.sum(delta, (0, 2, 3))
		return delta_out
	    ############################################################################
