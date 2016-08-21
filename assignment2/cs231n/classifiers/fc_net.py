import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
   
    W1=np.random.randn(input_dim,hidden_dim)*weight_scale  #hidden_dim,input_dim
    b1=np.zeros(hidden_dim) 
    W2=np.random.randn(hidden_dim,num_classes)*weight_scale #num_classes,hidden_dim
    b2=np.zeros(num_classes) 
    self.params['W1']=W1
    self.params['W2']=W2
    self.params['b1']=b1
    self.params['b2']=b2

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    #pass
    W1=self.params['W1']
    W2=self.params['W2']
    b1=self.params['b1']
    b2=self.params['b2']
    #X=np.reshape(X,(X.shape[0],-1)) #ADD 2016.7.19
    #print "X.shape:",X.shape,"W1.shape:",W1.shape,"b1.shape:",b1.shape
    out_l1, cache_l1=affine_relu_forward(X, W1, b1)
    out_l2, cache_l2=affine_forward(out_l1, W2, b2) #affine_relu_forward
    scores=out_l2
    #print "scores:",scores
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    #pass
    loss_data,grads_softmax = softmax_loss(scores, y)
    loss_reg=0.5* self.reg*(np.sum(W1*W1)+np.sum(W2*W2))
    loss=loss_data+loss_reg  
    
    #grads=grads_data+grads_reg
    dx2, dw2, db2=affine_backward(grads_softmax, cache_l2) #affine_relu_backward
    grads['W2']=dw2+self.reg*W2 
    grads['b2']=db2
    
    dx1, dw1, db1=affine_relu_backward(dx2,cache_l1)
    grads['W1']=dw1+self.reg*W1
    grads['b1']=db1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer
    giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    self.cacheee={}
    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    #pass
    #print "self.num_layers:",self.num_layers
    for i in xrange(self.num_layers):
        if i==0:                    #input layer
            self.params['W1']=np.random.randn(input_dim,hidden_dims[0])*weight_scale
            self.params['b1']=np.zeros(hidden_dims[0]) 
            if self.use_batchnorm:
                self.params['gamma'+str(i+1)]=np.ones(hidden_dims[i]) #initial gamma
                self.params['beta'+str(i+1)]=np.zeros(hidden_dims[i]) #initial beta
        elif i==(self.num_layers-1):#output layer
            self.params['W'+str(self.num_layers)]=np.random.randn(hidden_dims[i-1],num_classes)*weight_scale #W3--->W+self.num_layers
            self.params['b'+str(self.num_layers)]=np.zeros(num_classes) 
        else:                      #hidden layer #i==1
            self.params['W'+str(i+1)]=np.random.randn(hidden_dims[i-1],hidden_dims[i])*weight_scale #W2
            self.params['b'+str(i+1)]=np.zeros(hidden_dims[i]) #b2
            if self.use_batchnorm:
                self.params['gamma'+str(i+1)]=np.ones(hidden_dims[i]) #initial gamma
                self.params['beta'+str(i+1)]=np.zeros(hidden_dims[i]) #initial beta
            
    #i==0
    #W1=np.random.randn(input_dim,hidden_dims[0])*weight_scale  #hidden_dim,input_dim
    #b1=np.zeros(hidden_dims[0]) 
     
    #i==1
    #W2=np.random.randn(hidden_dims[0],hidden_dims[1])*weight_scale
    #b2=np.zeros(hidden_dims[1])
     
    #i==2
    #W3=np.random.randn(hidden_dims[1],num_classes)*weight_scale #num_classes,hidden_dim
    #b3=np.zeros(num_classes) 
    
    #self.params['W1']=W1
    #self.params['b1']=b1
    #self.params['W2']=W2
    #self.params['b2']=b2
    #self.params['W3']=W3
    #self.params['b3']=b3
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    
  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    #pass
    for i in xrange(self.num_layers):
        ww='W'+str(i+1)
        ww=self.params[ww] #W1=self.params['W1']
        bb='b'+str(i+1)
        bb=self.params[bb] #W2=self.params['W2']
        #print "i:",i
        if i==0: #first affline layer                    
           ##################################################################################################################################
           if self.use_batchnorm:                   #----------------------------------->ADD DROPOUT 2016.07.23
              outt,cachee=affline_batchnorm_relu_forward(X,ww,bb,self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i]) #TODO ADD 2016.08.02
           ######################################################################################batchnorm_forward###########################
           else:
               outt,cachee=affine_relu_forward(X,ww,bb) ##1.out_l1, cache_l1=affine_relu_forward(X, W1, b1)--->2.outt,cachee=affine_relu_forward(X,ww,bb)-----> 
           ##################################################################################################################################
           if self.use_dropout:                     #----------------------------------->ADD DROPOUT 2016.07.23
               outt, cache_droput=dropout_forward(outt, self.dropout_param)
               self.cacheee['cache_droput'+str(i+1)]=cache_droput ##save cache_dropout
           ####################################################################################################################################
           outt_previous=outt
           self.cacheee['cache'+str(i+1)]=cachee #save cache for compute gradient
        elif i!=self.num_layers-1:#middle affline layers #------------------------------->ADD DROPUT 2016.07.23
           if self.use_batchnorm:
               outt,cachee=affline_batchnorm_relu_forward(outt_previous,ww,bb,self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i]) 
           else:
               outt,cachee=affine_relu_forward(outt_previous,ww,bb) #1.out_l2, cache_l2=affine_relu_forward(out_l1, W2, b2) --->2.outt,cachee=affine_relu_forward(X,ww,bb)----->    
           ###################################################################################################################################
           if self.use_dropout:
               outt, cache_droput=dropout_forward(outt, self.dropout_param)
               self.cacheee['cache_droput'+str(i+1)]=cache_droput #save cache_dropout TODO
           ###################################################################################################################################
           outt_previous=outt
           self.cacheee['cache'+str(i+1)]=cachee #save cache for compute gradient
        else:#last affline layer
           outt,cachee=affine_forward(outt_previous,ww,bb) #out_l3, cache_l3=affine_forward(out_l2, W3, b3)
           self.cacheee['cache'+str(i+1)]=cachee #save cache for compute gradient
    scores=outt   
    #print "scores.shape:",scores.shape
    #W1=self.params['W1']
    #W2=self.params['W2']
    #b1=self.params['b1']
    #b2=self.params['b2']
    #W3=self.params['W3']
    #b3=self.params['b3']
    
    #out_l1, cache_l1=affine_relu_forward(X, W1, b1)
    #out_l2, cache_l2=affine_relu_forward(out_l1, W2, b2)  
    #out_l3, cache_l3=affine_forward(out_l2, W3, b3)  
    #scores=out_l3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    #pass
    loss_data,grads_softmax = softmax_loss(scores, y)
    
    #following is equals to--------------------->loss_reg=0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)++np.sum(W3*W3))
    sum=0
    for i in xrange(self.num_layers):
        ww='W'+str(i+1)
        ww=self.params[ww] #W1=self.params['W1']
        sum+=np.sum(ww*ww)
    loss_reg=0.5*self.reg*sum #equals to ----->loss_reg=0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)++np.sum(W3*W3))
    loss=loss_data+loss_reg  
    
    #grads=grads_data+grads_reg
    #dx3, dw3, db3=affine_backward(grads_softmax, cache_l3) 
    #grads['W3']=dw3+self.reg*W3
    #grads['b3']=db3
    
    #dx2, dw2, db2=affine_relu_backward(dx3, cache_l2)
    #grads['W2']=dw2+self.reg*W2 
    #grads['b2']=db2
    
    #dx1, dw1, db1=affine_relu_backward(dx2,cache_l1)
    #grads['W1']=dw1+self.reg*W1
    #grads['b1']=db1
    #dx_last='dx'+str(self.num_layers)
    #dw_last='dw'+str(self.num_layers)
    #db_last='db'+str(self.num_layers)
    
    i=self.num_layers#first end to first.
    while i>0:
        #dxi='dx'+str(i)
        #dwi='dw'+str(i)
        #dbi='db'+str(i)
        if i==self.num_layers:#last affline layer. ---->e.g. i==3
            dxi, dwi, dbi=affine_backward(grads_softmax, self.cacheee['cache'+str(i)]) #dx3, dw3, db3=affine_backward(grads_softmax, cache_l3) ###########################
            dx_parent=dxi
            dw_parent=dwi
            db_parent=dbi
            grads['W'+str(i)]=dwi+self.reg*self.params['W'+str(i)]
            grads['b'+str(i)]=dbi
            
        else:#affline relu layer.---->e.g.i==2  
            if self.use_batchnorm:
                dxi, dwi, dbi,dgamma,dbeta=affline_batchnorm_relu_backward(dx_parent,self.cacheee['cache'+str(i)]) #ADD BATCH NORM 2016.08.02
                grads['gamma'+str(i)]=dgamma 
                grads['beta'+str(i)]=dbeta
                #update gamma & beta
                #self.params['gamma'+str(i+1)]-=
            else:
                dxi, dwi, dbi=affine_relu_backward(dx_parent,self.cacheee['cache'+str(i)]) ##dx1, dw1, db1=affine_relu_backward(dx2,cache_l1) ##################################################      
            ####################################################################################################################################################################################
            if self.use_dropout:
                dx_parent=dropout_backward(dxi, self.cacheee['cache_droput'+str(i)])  #ADD DROPOUT 2016.07.23
            ####################################################################################################################################################################################
            dx_parent=dxi
            dw_parent=dwi
            db_parent=dbi
            #print "dwi.shape:",dwi.shape,"self.params['W'+str(i)].shape:",self.params['W'+str(i)].shape
            grads['W'+str(i)]=dwi+self.reg*self.params['W'+str(i)] ################### TODO TODO TODO TODO TODO TODO TODO TODO
            grads['b'+str(i)]=dbi
             
           
        i=i-1
         
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
