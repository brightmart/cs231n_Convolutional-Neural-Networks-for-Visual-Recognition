import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  #print("num_classes:",range(num_classes)) #('num_classes:', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):#for each row
      #print("X[i].shape:",X[i].shape)
      #print("W.shape:",W.shape)
      scores = X[i].dot(W) #'scores:', array([-0.18192299,  0.08821669, -1.22839927, -0.22183938,  0.32180663,-0.31810879,  0.47696505,  0.35287976, -0.05049794, -0.35763363]))
      correct_class_score = scores[y[i]] #y[i] is a lable,value can be 0,1,2,.... scores[y[i]], value can be: scores[2]
      for j in xrange(num_classes):#for each column
          if j == y[i]:#if it is the index for correct class, not count its loss.
              continue
          margin = scores[j] - correct_class_score + 1 # note delta = 1
          if margin > 0:
            loss += margin
            #dW[i,j]=X[i,j] #for j not equal to yi
            #the value of dW at jth column---->accumulated X value at ith row,transpose
            dW[:, j] += X[i, :].T # sums each contribution of the x_i's (for the jth column of all data,jth not equal to yi)           
            dW[:, y[i]] -= X[i, :].T # this is really a sum over j != y_i.(for the y[i]th colum of all data,j equal to yi)
            #dW[i,y[i]]=-1.0*rowMarginCount*X[i,y[i]]
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss. #---->1/2*reg*W*W
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################         
  dW /= num_train
  dW += reg * W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #full vectorized
  #num_classes = W.shape[1]
  num_train = X.shape[0]
  scores=X.dot(W)#1.get scores matrix .scores belong to---->(N,D)...............#print("X.shape:",X.shape,"W.shape:",W.shape,"scores.shape",scores.shape)
  scores_correct=scores[np.arange(num_train),y] # 2.get target scores
  scores_correct_reshaped=np.reshape(scores_correct,(len(scores_correct),1)) #3.reshape for subtraction
  margin=np.maximum(0,scores-scores_correct_reshaped+1) #4.get individual loss
  margin[np.arange(num_train),y]=0 #5.set value as 0 for those target score
  lossSum=np.sum(margin,axis=1) #6.sum up --->>--->>--horizontally---->
  lossNormalize=np.sum(lossSum)/(num_train) #7.get loss
  loss_reg= 0.5 * reg * np.sum(W * W) #8.count regularization term
  loss=lossNormalize+loss_reg
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #1. set margin to 0 or scalr before operation
  binary=margin #matrix for binary operation
  binary[margin>0]=1
  binary[margin<=0]=0
  binary_row_count_positive=np.sum(binary>0,axis=1)
  binary[np.arange(num_train),y]=-1.0*binary_row_count_positive[np.arange(num_train)]
  #2.set dW for all
  dW=(X.T).dot(binary) 
  dW=dW/(num_train)
  #add dW regularization term
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

  #half vectorized OF LOSS
  #num_train=X.shape[0]
  #loss=0
  #for i in xrange(num_train):
  #    scores=X[i,:].dot(W)
  #    margins=scores-scores[y[i]]+1
  #    margins[y[i]]=0
  #    loss=loss+np.sum(margins)
  #loss=loss/num_train
  #loss_reg=0.5*reg*np.sum(W*W)
  #loss=loss+loss_reg