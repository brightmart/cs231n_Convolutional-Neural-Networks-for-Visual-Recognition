from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def affline_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):  
   """
  Convenience layer that perorms an affine transform followed by a batchnorm, then ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
 
  """
   #print "affline_batchnorm_relu_forward...."
   #print "x.shape:",x.shape,"w.shape:",w.shape
   a, fc_cache=affine_forward(x, w, b)
   #print "a.shape:",a.shape
   b, batchnorm_cache=batchnorm_forward(a, gamma, beta, bn_param)
   out, relu_cache = relu_forward(b)
   cache = (fc_cache, batchnorm_cache,relu_cache)
   return out, cache
  
def affline_batchnorm_relu_backward(dout, cache):
   """
   Backward pass for the affine-batchnorm-ReLU convenience layer
   
   """
   #print "affline_batchnorm_relu_backward...."
   fc_cache, batchnorm_cache,relu_cache=cache
   da = relu_backward(dout, relu_cache)
   dx_batchnorm, dgamma, dbeta=batchnorm_backward(da, batchnorm_cache)
   dx, dw, db = affine_backward(dx_batchnorm, fc_cache)
   return dx,dw,db,dgamma,dbeta
   

def conv_batchnorm_relu_forward(x, w, b, conv_param,gamma, beta, bn_param): #add 2016.8.4 used by batch normailization layer for conv net 
   """
  Convenience layer that perorms an conv transform followed by a batchnorm, then ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
 
  """
   #1.conv_forward 
   out_conv, cache_conv = conv_forward_fast(x, w, b, conv_param)
   #2.batchnorm foward for conv
   out_batchnorm, cache_batchnorm=spatial_batchnorm_forward(out_conv, gamma, beta, bn_param)
   #3.relu_forward
   out,cache_relu=relu_forward(out_batchnorm)
   cache=(cache_conv,cache_batchnorm,cache_relu)
   return out, cache
   
def conv_batch_relu_backward(dout, cache):
    cache_conv,cache_batchnorm,cache_relu=cache
    #1.relu backward
    dx_relu = relu_backward(dout, cache_relu)
    #2.batchnormal backward
    dx_batchnorm, dgamma, dbeta = spatial_batchnorm_backward(dx_relu, cache_batchnorm)
    #3.conv backward
    dx, dw, db = conv_backward_fast(dx_batchnorm, cache_conv)
    return dx,dw,db,dgamma,dbeta
    
    
def conv_batchnorm_relu_pool_forward(x, w, b, conv_param,pool_param,gamma, beta, bn_param): #add 2016.8.4 used by batch normailization layer for conv net 
   """
  Convenience layer that perorms an conv transform followed by a batchnorm, then ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
 
  """
   #1.conv_forward 
   out_conv, cache_conv = conv_forward_fast(x, w, b, conv_param)
   #2.batchnorm foward for conv
   out_batchnorm, cache_batchnorm=spatial_batchnorm_forward(out_conv, gamma, beta, bn_param)
   #3.relu_forward
   out_relu,cache_relu=relu_forward(out_batchnorm)
   #4.pool_forward
   out, cache_pool = max_pool_forward_fast(out_relu, pool_param)
   cache=(cache_conv,cache_batchnorm,cache_relu,cache_pool)
   return out, cache

def conv_batchnorm_relu_pool_backward(dout, cache):
    (cache_conv,cache_batchnorm,cache_relu,cache_pool)=cache
    #1.pool
    dx_pool=max_pool_backward_fast(dout, cache_pool)
    #2.relu
    dx_relu = relu_backward(dx_pool, cache_relu)
    #3.batchnorm
    dx_batchnorm, dgamma, dbeta = spatial_batchnorm_backward(dx_relu, cache_batchnorm)
    #4.conv
    dx, dw, db = conv_backward_fast(dx_batchnorm, cache_conv)
    return dx,dw,db,dgamma,dbeta 
    