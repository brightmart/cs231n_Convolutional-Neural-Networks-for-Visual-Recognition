import numpy as np
from random import shuffle

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
  #pass
  num_train=X.shape[0]
  #num_classes=W.shape[1]
  for i in xrange(num_train):
      scores_i=X[i].dot(W)
      scores_i-=np.max(scores_i)
      scores_correct_label=scores_i[y[i]]
      #1.loss
      probabilities=np.exp(scores_correct_label)/(np.sum(np.exp(scores_i)))
      loss_i=-1.0*(np.log(probabilities))
      loss+=loss_i
      
      #2.gradient
      #probabilities_all_i=((np.exp(scores_i)).T/(np.sum(np.exp(scores_i)))).T
      #binary_matrix_i=np.zeros((scores_i.shape))#O.K.
      #binary_matrix_i[y[i]]=1#O.K.
      #print("X[i]:",X[i].shape)
      #print("binary_matrix_i:",binary_matrix_i)
      #print("probabilities_all_i:",probabilities_all_i)
      #dW[i]=-1.0*X[i].T.dot(binary_matrix_i-probabilities_all_i)
      
      #for j in xrange(num_classes):
          #print("X[i,j]:",X[i,j]) #
       #   dW[i,j]=-1.0*X[i,j]*(binary_matrix_i[j]-probabilities_all_i[j]) #TODO
          #if j!=-1:
             # print("dW[i,j]:",dW[i,j])
          #if j==y[i]:
          #    dW[i,y[i]]=-1.0*X[i,j]*(binary_matrix_i[j]-probabilities_all_i[j])
              #dW[i,y[i]]=-1.0/(num_train)*(np.log(np.sum(np.exp(scores_i))))*X[i,y[i]]
          #else:
              #dW[i,j]=-1.0*X[i,j]+np.log(np.sum(np.exp(scores_i)))*X[i,j]
      
         
  loss=loss/(num_train)  
  loss_reg=0.5*reg*np.sum(W*W)
  loss+=loss_reg
  
  #############################################################################
  #gradient
  #1.scores
  scores=X.dot(W) #(N,C)
  #2.normalized scores:subtract mean
  scores=(scores.T-np.max(scores,axis=1)).T
  binary_matrix=np.zeros((scores.shape))#O.K.
  #2.set cell value to 1 if yi=j(correct label)
  binary_matrix[xrange(num_train),y]=1#O.K.
  #3.calculate possiblity for all scores.
  possibility_all=((np.exp(scores)).T/(np.sum(np.exp(scores),axis=1))).T
  #4.calucate gradient
  dW=(-1.0/num_train)*X.T.dot(binary_matrix-possibility_all)+reg*W
  ##############################################################################
      
           
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW



def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  #1.########################vectorized loss###################################
  num_train=X.shape[0]
  #num_classes=W.shape[1]
  #1.get scores
  scores=X.dot(W) #(N,C)
  #2.normalized scores:subtract mean
  scores=(scores.T-np.max(scores,axis=1)).T
  #3.correct class's scores
  scores_correct_class=scores[xrange(num_train),y]
  #4.calculate normalized possiblity
  possibility=(np.exp(scores_correct_class))/(np.sum(np.exp(scores),axis=1))
  #5.loss vector, each element is a loss for row in scores matrix
  loss=-1*np.log(possibility)
  #6.accumulate loss
  loss=np.sum(loss)
  #7.normalized loss
  loss_data=loss/num_train
  #8.calculate regularization term
  loss_reg=0.5*reg*np.sum(W*W)
  #9.final loss
  loss=loss_data+loss_reg
  
  #2.####################vectorized gradient###################################
  #1.init all zero matrix
  binary_matrix=np.zeros((scores.shape))#O.K.
  #2.set cell value to 1 if yi=j(correct label)
  binary_matrix[xrange(num_train),y]=1#O.K.
  #3.calculate possiblity for all scores.
  possibility_all=((np.exp(scores)).T/(np.sum(np.exp(scores),axis=1))).T
  #4.calucate gradient
  dW=(-1.0/num_train)*X.T.dot(binary_matrix-possibility_all)+reg*W
  #print("dW.shape:",dW.shape)

  return loss, dW

