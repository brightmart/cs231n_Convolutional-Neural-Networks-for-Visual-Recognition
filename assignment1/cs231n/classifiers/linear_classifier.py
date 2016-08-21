import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *

class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    #if self.W is None:
    # lazily initialize W---->always initialize)
    self.W = 0.001 * np.random.randn(dim, num_classes)*np.sqrt(dim/2.0) 

    # Run stochastic gradient descent to optimize W
    loss_history = []
	#Try using momentum update
    v=0 #init to zero
    mu=0.5 #int to 0.5, and increase it later.
    for it in xrange(num_iters):
	  if num_iters%100==0:
		  mu+=0.05
          #if num_iters>=1500:
          #    learning_rate*=0.7
          if mu>=0.99:
			 mu=0.99
    X_batch = None
    y_batch = None
      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      #pass
      #1. get the batch for this iteration.
    batch_indices=np.random.choice(num_train,batch_size,replace=True)
    X_batch=X[batch_indices] #('X_batch.shape:', (200L, 3073L))
      #print("X_batch.shape:",X_batch.shape) 
    y_batch=y[batch_indices] #('y_batch.shape:', 200)
      #print("y_batch.shape:",len(y_batch))
	  #loss_vectorized, grad_vectorized = svm_loss_vectorized(self.W, X_batch, y_batch, reg)
	  #self.W+=-learning_rate*grad_vectorized
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      #2. evaluate loss and gradient
    loss, grad = self.loss(X_batch, y_batch, reg)
    loss_history.append(loss)

      #3. perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      #pass
      #self.W-=learning_rate*grad
    v=mu*v-learning_rate*grad
    self.W+=v
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

    if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: D x N array of training data. Each column is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[1])
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    #pass
    scores=X.dot(self.W) #1.get scores
    y_pred=np.argmax(scores,axis=1) #2.find the index for highest value in the row
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    pass


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

