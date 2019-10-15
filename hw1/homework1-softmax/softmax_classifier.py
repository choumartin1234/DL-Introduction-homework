import numpy as np

def softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    N = input.shape[0]
    exp = np.exp(np.matmul(input, W))   
    h = exp / exp.sum(axis = 1)[:, np.newaxis]      # softmax function, add a new dimension to broadcast
    cross_entropy = (label * np.log(h)).sum() / N
    weight_decay = lamda * (W**2).sum() / 2        # for regularization
    loss = -1*cross_entropy + weight_decay 
    gradient = -1*(np.matmul(input.T, label-h))/N+lamda*W  # remember weight decay
    prediction = h.argmax(axis = 1)
    ############################################################################

    return loss, gradient, prediction
