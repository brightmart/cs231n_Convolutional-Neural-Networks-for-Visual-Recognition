import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W=input_dim
    #H_after_conv=(H-filter_size)+1
    H_after_pool=H/2
    
  #- x: Input data of shape (N, C, H, W)
  #- w: Filter weights of shape (F, C, HH, WW)
  #- b: Biases, of shape (F,)
  
    W1=np.random.randn(num_filters,C,filter_size,filter_size)*weight_scale  
    b1=np.random.randn(num_filters)*weight_scale 
    self.params['W1']=W1
    self.params['b1']=b1
    
    W2=np.random.randn(H_after_pool*H_after_pool*num_filters,hidden_dim)*weight_scale  
    b2=np.random.randn(hidden_dim)*weight_scale 
    self.params['W2']=W2
    self.params['b2']=b2
    
    W3=np.random.randn(hidden_dim,num_classes)*weight_scale  
    b3=np.random.randn(num_classes)*weight_scale 
    self.params['W3']=W3
    self.params['b3']=b3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2] 
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #pass
    out_l1,cache_l1=conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
    out_l2,cache_l2=affine_relu_forward(out_l1, W2, b2)
    out_l3,cache_l3=affine_forward(out_l2, W3, b3)
    scores=out_l3
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    #pass
    loss_data,grads_softmax = softmax_loss(scores,y)
    loss_reg=0.5* self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))
    loss=loss_data+loss_reg  
    
    #grads=grads_data+grads_reg
    dx3, dw3, db3=affine_backward(grads_softmax, cache_l3) 
    grads['W3']=dw3+self.reg*W3 
    grads['b3']=db3
    
    dx2, dw2, db2=affine_relu_backward(dx3,cache_l2)
    grads['W2']=dw2+self.reg*W2
    grads['b2']=db2
    
    dx1, dw1, db1 = conv_relu_pool_backward(dx2, cache_l1)
    grads['W1']=dw1+self.reg*W1
    grads['b1']=db1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
class DeepLayerConvNet(object):
  """
  A more deep than three-layer convolutional network with the following architecture:
  
  [conv-relu-conv-relu-pool]x2 - [affine]x2 - [softmax or SVM] equals to
  1.[conv-relu]-2.[conv-relu-pool--3.[conv-relu]-4.[conv-relu-pool] - 5.[affine-relu] -6.[affine] - [softmax] 
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32,use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm #Add 2016.08.04
    
    ############################################################################
    # TODO: Initialize weights and biases for the multi-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W=input_dim
    #Before adding BN: 1.[conv-relu]-   2.[conv-relu-pool--   3.[conv-relu]-   4.[conv-relu-pool] -    5.[affine-relu] -   6.[affine] - [softmax] 
    #After  adding BN: 1.[conv-BN-relu]-2.[conv-BN-relu-pool--3.[conv-BN-relu]-4.[conv-BN-relu-pool] - 5.[affine-BN-relu] -6.[affine] - [softmax] 
    W1=np.random.randn(num_filters,C,filter_size,filter_size)*weight_scale  
    b1=np.random.randn(num_filters)*weight_scale 
    self.params['W1']=W1
    self.params['b1']=b1
    if self.use_batchnorm:
        self.params['gamma'+str(0+1)]=np.ones(num_filters) #initial gamma
        self.params['beta'+str(0+1)]=np.zeros(num_filters) #initial beta
    
    W2=np.random.randn(num_filters,num_filters,filter_size,filter_size)*weight_scale  
    b2=np.random.randn(num_filters)*weight_scale 
    self.params['W2']=W2
    self.params['b2']=b2
    if self.use_batchnorm:
        self.params['gamma'+str(1+1)]=np.ones(num_filters) #initial gamma
        self.params['beta'+str(1+1)]=np.zeros(num_filters) #initial beta
    
    W3=np.random.randn(num_filters,num_filters,filter_size,filter_size)*weight_scale   #num_filters,num_filters,filter_size,filter_size
    b3=np.random.randn(num_filters)*weight_scale 
    self.params['W3']=W3
    self.params['b3']=b3
    if self.use_batchnorm:
        self.params['gamma'+str(2+1)]=np.ones(num_filters) #initial gamma
        self.params['beta'+str(2+1)]=np.zeros(num_filters) #initial beta
    
    W4=np.random.randn(num_filters,num_filters,filter_size,filter_size)*weight_scale  
    b4=np.random.randn(num_filters)*weight_scale 
    self.params['W4']=W4
    self.params['b4']=b4
    if self.use_batchnorm:
        self.params['gamma'+str(3+1)]=np.ones(num_filters) #initial gamma
        self.params['beta'+str(3+1)]=np.zeros(num_filters) #initial beta
    
    W5=np.random.randn(H/4*H/4*num_filters,hidden_dim)*weight_scale  #H/2*H/2
    b5=np.random.randn(hidden_dim)*weight_scale 
    self.params['W5']=W5
    self.params['b5']=b5
    if self.use_batchnorm:
        self.params['gamma'+str(4+1)]=np.ones(hidden_dim) #initial gamma
        self.params['beta'+str(4+1)]=np.zeros(hidden_dim) #initial beta
    
    W6=np.random.randn(hidden_dim,num_classes)*weight_scale  
    b6=np.random.randn(num_classes)*weight_scale 
    self.params['W6']=W6
    self.params['b6']=b6
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(100)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    print "initied...."
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    
    ##After  adding BN: 1.[conv-BN-relu]-2.[conv-BN-relu-pool--3.[conv-BN-relu]-4.[conv-BN-relu-pool] - 5.[affine-BN-relu] -6.[affine] - [softmax] 
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2] 
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    if self.use_batchnorm: #layer1
        out_l1,cache_l1=conv_batchnorm_relu_forward(X, W1, b1, conv_param,self.params['gamma'+str(0+1)], self.params['beta'+str(0+1)], self.bn_params[0])
        #print "VERY GOOD! pass first layer."
    else:
        out_l1,cache_l1=conv_relu_forward(X, W1, b1, conv_param)    #conv_relu_forward(X, W1, b1, conv_param) -->conv_batchnorm_relu_forward
    
    if self.use_batchnorm: #layer2
        out_l2,cache_l2=conv_batchnorm_relu_pool_forward(out_l1, W2, b2, conv_param,pool_param,self.params['gamma'+str(1+1)], self.params['beta'+str(1+1)], self.bn_params[1])
    else:
        out_l2,cache_l2=conv_relu_pool_forward(out_l1, W2, b2, conv_param,pool_param) #-------------------> conv_relu_pool_forward(x, w, b, conv_param, pool_param) 
    
    if self.use_batchnorm: #layer3
        out_l3,cache_l3=conv_batchnorm_relu_forward(out_l2, W3, b3, conv_param,self.params['gamma'+str(2+1)], self.params['beta'+str(2+1)], self.bn_params[2])
    else:
        out_l3,cache_l3=conv_relu_forward(out_l2, W3, b3, conv_param)
    
    if self.use_batchnorm: #layer4
        out_l4,cache_l4=conv_batchnorm_relu_pool_forward(out_l3, W4, b4, conv_param,pool_param,self.params['gamma'+str(3+1)], self.params['beta'+str(3+1)], self.bn_params[3])
    else:
        out_l4,cache_l4=conv_relu_pool_forward(out_l3, W4, b4, conv_param,pool_param) #-------------------> conv_relu_pool_forward(x, w, b, conv_param, pool_param)   
    
    if self.use_batchnorm: #layer5
        out_l5,cache_l5=affline_batchnorm_relu_forward(out_l4, W5, b5,self.params['gamma'+str(4+1)], self.params['beta'+str(4+1)], self.bn_params[4])
    else:
        out_l5,cache_l5=affine_relu_forward(out_l4, W5, b5)
    
    out_l6,cache_l6=affine_forward(out_l5, W6, b6) #do not attach Batch normalization to last layer
    scores=out_l6
    #print "WOW.forward almost done!"
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################ 
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    #After adding BN: 1.[conv-BN-relu]-2.[conv-BN-relu-pool--3.[conv-BN-relu]-4.[conv-BN-relu-pool] - 5.[affine-BN-relu] -6.[affine] - [softmax] 
    loss_data,grads_softmax = softmax_loss(scores,y)
    loss_reg=0.5* self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3)+np.sum(W4*W4)+np.sum(W5*W5)+np.sum(W6*W6))
    loss=loss_data+loss_reg  
    
    #grads=grads_data+grads_reg
    dx6, dw6, db6=affine_backward(grads_softmax, cache_l6) 
    grads['W6']=dw6+self.reg*W6
    grads['b6']=db6
    
    if self.use_batchnorm:#layer5
        dx5, dw5, db5,dgamma,dbeta=affline_batchnorm_relu_backward(dx6,cache_l5)
        grads['gamma'+str(5)]=dgamma 
        grads['beta'+str(5)]=dbeta
    else:
        dx5, dw5, db5=affine_relu_backward(dx6,cache_l5)
    grads['W5']=dw5+self.reg*W5
    grads['b5']=db5
    
    if self.use_batchnorm:#layer4
        dx4,dw4,db4,dgamma,dbeta=conv_batchnorm_relu_pool_backward(dx5, cache_l4)  
        grads['gamma'+str(4)]=dgamma 
        grads['beta'+str(4)]=dbeta
    else:
        dx4, dw4, db4 = conv_relu_pool_backward(dx5, cache_l4) #conv_relu_pool_backward(dout, cache)
    grads['W4']=dw4+self.reg*W4
    grads['b4']=db4
    
    if self.use_batchnorm:#layer3
        dx3, dw3, db3,dgamma,dbeta=conv_batch_relu_backward(dx4, cache_l3)
        grads['gamma'+str(3)]=dgamma 
        grads['beta'+str(3)]=dbeta
    else:
        dx3, dw3, db3 = conv_relu_backward(dx4, cache_l3)
    grads['W3']=dw3+self.reg*W3
    grads['b3']=db3
    
    if self.use_batchnorm:#layer2
        dx2, dw2, db2 ,dgamma,dbeta=conv_batchnorm_relu_pool_backward(dx3, cache_l2)  
        grads['gamma'+str(2)]=dgamma 
        grads['beta'+str(2)]=dbeta
    else:
        dx2, dw2, db2 = conv_relu_pool_backward(dx3, cache_l2) 
    grads['W2']=dw2+self.reg*W2
    grads['b2']=db2
    
    if self.use_batchnorm: #layer1
        dx1, dw1, db1,dgamma,dbeta=conv_batch_relu_backward(dx2, cache_l1)
        grads['gamma'+str(1)]=dgamma 
        grads['beta'+str(1)]=dbeta
    else:
        dx1, dw1, db1 = conv_relu_backward(dx2, cache_l1)
    grads['W1']=dw1+self.reg*W1
    grads['b1']=db1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
    
pass