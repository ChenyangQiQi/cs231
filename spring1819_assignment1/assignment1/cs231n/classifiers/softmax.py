from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W) #(N,C)
    scores = scores - np.amax(scores, axis=1).reshape(-1,1)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    softmax = np.zeros(num_train) 
    for i in range(num_train):
      softmax[i] = np.exp(scores[i, y[i]]) / np.sum(np.exp(scores[i]))
    loss += -np.sum(np.log(softmax))
    loss /= num_train
    loss += reg*np.sum(np.square(W))
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #print("softmax shape", softmax.shape)
    #print("x[i]", X[i].shape)
    #print("Denominator", (np.exp(X[i].dot(W[:,y[i]]))**2).shape)
    for i in range(num_train):
      for j in range(num_classes):
        softmax[i] = np.exp(scores[i,j]) / np.sum(np.exp(scores[i]), axis=0)
        if j == y[i]:
          dW[:, j] += (-1 + softmax[i]) * X[i]
        else:
          dW[:, j] += softmax[i] * X[i]
    dW /= num_train
    dW += 2*reg*W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    scores_shifted = scores - np.max(scores, axis=1).reshape(-1,1)
    softmax = np.exp(scores_shifted) / np.sum(np.exp(scores_shifted), axis=1).reshape(-1,1)
    loss = -1.0* np.sum(np.log(softmax[range(num_train), y]))
    loss /= num_train
    loss += reg*np.sum(W**2)
    
    softmax[range(num_train), y] -= 1
    dW = (X.T).dot(softmax)
    dW /= num_train
    dW += 2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
