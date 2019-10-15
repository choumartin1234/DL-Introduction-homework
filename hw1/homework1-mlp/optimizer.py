""" Optimizer Class """

import numpy as np

class SGD():
    def __init__(self, learningRate, weightDecay):
        self.learningRate = learningRate
        self.weightDecay = weightDecay

    # One backpropagation step, update weights layer by layer
    def step(self, model):
        layers = model.layerList
        for layer in layers:
            if layer.trainable:
                ############################################################################
                    # TODO: Put your code here
                # Calculate diff_W and diff_b using layer.grad_W and layer.grad_b.
                # Do not forget the weightDecay term.
                momentum = 0.5 # apply momentum method for better result
                # W = W - eta * (grad + lambda * W) + momentum * diff_W
                layer.diff_W = -1 * self.learningRate * (layer.grad_W + self.weightDecay * layer.W) + layer.diff_W * momentum
                layer.diff_b = -1 * self.learningRate * (layer.grad_b + self.weightDecay * layer.b) + layer.diff_b * momentum
                ############################################################################

                # Weight update
                layer.W += layer.diff_W
                layer.b += layer.diff_b
