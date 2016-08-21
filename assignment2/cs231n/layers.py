import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  #reshape X
  x_=np.reshape(x,(x.shape[0],-1))
  #computer dot product
  out=np.dot(x_,w)+b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  #print "cache[0]:",cache[0].shape,cache[1].shape
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx=dout.dot(w.T) #gradient for x
  dx=np.reshape(dx,x.shape)
  
  x_=np.reshape(x,(x.shape[0],-1))
  dw=x_.T.dot(dout) #gradient for w
  
  ones=np.ones((dout.shape[0]))
  db=dout.T.dot(ones) #gradient for b--->#db shape should be:(M,),same as shape of b.
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  #out=x.copy()          #method 1
  #out[out<0]=0
  
  #out=x*np.float64(x>0) #method2 
  
                         #method 3 
  out=np.maximum(x,0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  #dx_local=np.ones((x.shape))
  #dx_local[x<0]=0
  #dx=dout*(dx_local)
  
  dx=dout.copy()#np.array(dout,copy=True)
  dx[x<0]=0
  
  #dx = dout * (x>0) 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    #pass
     sample_mean=np.mean(x,axis=0)
     sample_var=np.std(x,axis=0)
     X_normalized=(x-sample_mean)/np.sqrt(sample_var**2+eps)
     out=gamma*X_normalized+beta
     running_mean=momentum*running_mean+(1-momentum)*sample_mean
     running_var=momentum*running_var+(1-momentum)*sample_var
     #save intermediates to cache
     cache=(x,sample_mean,sample_var,gamma,beta,eps,X_normalized) #cache = x, X_normalized, gamma, running_mean, running_var, eps #TODO
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    #pass
    X_normalized=(x-running_mean)/running_var
    out=gamma*X_normalized+beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  #pass
  #print "dout.shape:",dout.shape
  #x,sample_mean,sample_var,gamma,beta=cache
  x,sample_mean,sample_var,gamma,beta,eps,X_normalized=cache
  m=dout.shape[0]
  #x, X_normalized, gamma, running_mean, running_var, eps=cache
  
  dx_normalized=dout*gamma
  dvariance_batch=np.sum((dx_normalized*(x-sample_mean)*(-0.5)*np.power(sample_var**2+eps,-1.5)),axis=0)
  dmean_batch=np.sum(dx_normalized*(-1.0/(np.sqrt(sample_var**2+eps))),axis=0)+dvariance_batch*(np.sum(-2.0*(x-sample_mean)/m,axis=0))
 
  dx=dx_normalized*(1.0/np.sqrt(sample_var**2+eps)) + dvariance_batch*(2.0*(x-sample_mean)/m)+dmean_batch*(1.0/m)
  dbeta=np.sum(dout,axis=0)
  dgamma=np.sum(dout*X_normalized,axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache): 
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  #pass
  #x, norm, gamma, mean, var, eps = cache
  
  #dbeta = np.sum(dout, axis=0)
  #dgamma = np.sum(dout * norm, axis=0)
  
  #dn = dout * gamma
  #invstd = (var + eps)**-.5
  #x -= mean
  #dx = invstd * (dn - np.mean(dn, 0)) - invstd**3 * x * np.mean(dn * x, 0)
  
  #  x,sample_mean,sample_var,gamma,beta,eps,X_normalized=cache
  #m=dout.shape[0]
  #x, X_normalized, gamma, running_mean, running_var, eps=cache
  x,sample_mean,sample_var,gamma,beta,eps,X_normalized=cache
  m=dout.shape[0]
  dx_normalized=dout*gamma
  dvariance_batch=np.sum((dx_normalized*(x-sample_mean)*(-0.5)*np.power(sample_var**2+eps,-1.5)),axis=0)
  dmean_batch=np.sum(dx_normalized*(-1.0/(np.sqrt(sample_var**2+eps))),axis=0)+dvariance_batch*(np.sum(-2.0*(x-sample_mean)/m,axis=0))
 
  dx=dx_normalized*(1.0/np.sqrt(sample_var**2+eps)) + dvariance_batch*(2.0*(x-sample_mean)/m)+dmean_batch*(1.0/m)
  dbeta=np.sum(dout,axis=0)
  dgamma=np.sum(dout*X_normalized,axis=0)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    #pass
    mask=(np.random.rand(*x.shape)<p)/p
    out=x*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    #pass
    out=x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    #pass
    dx=dout.copy()
    dx[mask==False]=0
    dx=dx/dropout_param['p']
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_vectorized_Mine(x, w, b, conv_param):#my vectorized implementation of conv forward.
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  F,C,HH,WW=w.shape
  N,C,H,W=x.shape
  stride=conv_param['stride']
  pad=conv_param['pad']
  
  H_=1+float(H+2*pad-HH)/float(stride) #H'
  W_=1+float(W+2*pad-WW)/float(stride) #W'
  out=np.zeros((N,F,H_,H_))
  #1.get x_col, with size (C*HH*WW,H'*W')
  #x_col=np.zeors(C*HH*WW,H_*W_)
  #2.get w_row, with size (F,C*HH*WW)
  #3.dot product of w_row with x_col to get result
  #4.reshape the result to shape of (H',W',F)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache
  
def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  #pass
  F,C,HH,WW=w.shape
  x_part=np.zeros((C,HH,WW)) #part of data
  N,C,H,W=x.shape
  stride=conv_param['stride']
  pad=conv_param['pad']
  
  H_=1+float(H+2*pad-HH)/float(stride) #H'
  W_=1+float(W+2*pad-WW)/float(stride) #W'
  out=np.zeros((N,F,H_,H_))
  current_width=0
  current_height=0  
  #x_pad=np.lib.pad(x,(pad,pad),'constant') #pad input
  x_pad=np.zeros((N,C,H+2*pad,W+2*pad))
  x_pad[:,:,pad:-pad,pad:-pad]=x #IMPORTANT---->should not use np.pad direct into x.
  #print "x_pad:",x_pad
  for i in xrange(N):#for each single input image
      for wi in xrange(F): #for each filter
          current_width=0
          current_height=0    
          for hh in xrange(np.int32(H_)):#height height height height
              current_height=stride*hh #IMPORTANT---->shift window
              if current_height<=H_:#NOT NECESSARY TO USE THIS CONDITION
                 for ww in xrange(np.int32(H_)):#width width width width 
                    current_width=stride*ww #IMPORTANT---->shift window
                    if current_width<=H_:#NOT NECESSARY TO USE THIS CONDITION
                        x_part=x_pad[i,0:C,current_height:current_height+HH,current_width:current_width+HH] #get part of data ------->x: Input data of shape (N, C, H, W) 
                        convolution=np.sum(x_part*w[wi])   #------------>w: Filter weights of shape (F, C, HH, WW)
                        out[i,wi,hh,ww]=convolution+b[wi] #save convolution(a single value) to an array
  #print "out:",out    
      
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive_(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  #pass
  x,w,b,conv_param=cache
  #print "dout.shape:",dout.shape,"w.shape:",w.shape #dout's shape should be same as out. and dx's shape should be....same as x.
  dx=np.zeros((x.shape))
  
  #dx=np.rot90(w).dot(dout)
  #dw=x.dot(dout)
  #db=dout
  #print "dout.shape:",dout.shape
  
  F,C,HH,WW=w.shape
  x_part=np.zeros((C,HH,WW)) #part of data
  N,C,H,W=x.shape
  stride=conv_param['stride']
  pad=conv_param['pad']
  
  H_=1+float(H+2*pad-HH)/float(stride) #H'
  W_=1+float(W+2*pad-WW)/float(stride) #W'
  out=np.zeros((N,F,H_,H_))
  current_width=0
  current_height=0  
  x_pad=np.zeros((N,C,H+2*pad,W+2*pad))
  x_pad[:,:,pad:-pad,pad:-pad]=x #IMPORTANT---->should not use np.pad direct into x.
  dx=np.zeros_like(x)
  for i in xrange(N):#for each single input
      for wi in xrange(F): #for each filter
          current_width=0
          current_height=0     
          for hh in xrange(np.int32(H_)):#height height height height
              current_height=stride*hh #IMPORTANT---->shift window
              for ww in xrange(np.int32(H_)):#width width width width 
                   current_width=stride*ww #IMPORTANT---->shift window
                   print "1.shape:",dx[i,:,current_height:current_height+HH,current_width:current_width+HH].shape,"2.shape",w[wi].shape,"3.shape:",dout[i,wi,hh,ww].shape
                   dx[i,:,current_height:current_height+HH,current_width:current_width+HH]+=w[wi]*dout[i,wi,hh,ww]
                          
                        #dx[p, :, row*stride:HH+row*stride, col*stride:WW+col*stride] += w[f]*dout[p, f, row, col]
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db
 
def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  x, w, b, conv_param = cache

  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  pad = conv_param['pad']
  stride = conv_param['stride']
  H_ = 1 + (H + 2 * pad - HH) / stride
  W_ = 1 + (W + 2 * pad - WW) / stride


  paddedX = np.zeros((N, C, H+2*pad, W+2*pad))
  paddedX[:, :, pad:-pad, pad:-pad] = x

  dx = np.zeros((N, C, H+2*pad, W+2*pad))
  dw = np.zeros(w.shape)

  for p in xrange(N):
    for f in xrange(F):
      for row in xrange(H_):
        for col in xrange(W_):
          dx[p, :, row*stride:HH+row*stride, col*stride:WW+col*stride] += w[f]*dout[p, f, row, col]
          dw[f] += paddedX[p, :, row*stride:HH+row*stride, col*stride:WW+col*stride]*dout[p, f, row, col]

  dx = dx[:, :, pad:-pad, pad:-pad]
  db = np.sum(dout, axis=(0, 2, 3))

  return dx, dw, db
  
def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  #pass
  N,C,H,W=x.shape
  pool_height=pool_param['pool_height']
  pool_width=pool_param['pool_width']
  stride=pool_param['stride']
  h_=((H-pool_height)/2.0)+1
  out=np.zeros((N,C,h_,h_))
  for p in xrange(N):#for each sample
      for c in xrange(C):#for each height*weight(2 dimensional)
          for row in xrange(np.int32(h_)):
              for col in xrange(np.int32(h_)):
                 x_part=x[p,c,stride*row:stride*row+pool_height,stride*col:stride*col+pool_width]
                 out[p,c,row,col]=np.max(x_part)
          
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x,pool_param=cache
  N,C,H,W=x.shape
  pool_height=pool_param['pool_height']
  pool_width=pool_param['pool_width']
  stride=pool_param['stride']
  h_=((H-pool_height)/2.0)+1
  dx=np.zeros_like(x)
  for p in xrange(N):#for each sample
      for c in xrange(C):#for each height*weight(2 dimensional)
          for row in xrange(np.int32(h_)):
              for col in xrange(np.int32(h_)):
                 x_part=x[p,c,stride*row:stride*row+pool_height,stride*col:stride*col+pool_width] #6.1 get part of x. x_shape---->(2,2)
                 x_part_flattened=x_part.flatten() #6.2 flattend the x_part
                 x_part_flattened2=np.zeros_like(x_part_flattened) #6.3 x_part_flattened2 has same shape as x_part_flattened, with all zero values
                 x_part_flattened2[x_part_flattened==np.max(x_part_flattened)]=1 #6.4 set the value of the element to 1 if it is the biggest element
                 x_part_flattened22=np.reshape(x_part_flattened2,[pool_height,pool_height]) #6.5 reshape back 
                 dx[p,c,stride*row:stride*row+pool_height,stride*col:stride*col+pool_width]=x_part_flattened22*dout[p,c,row,col] #6.6 assign value to dx,according to chain rule
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #print "dx:",dx
  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  #N, C, H, W= x.shape
  #for i in xrange(C):#x_reshaped=np.reshape(x,(N*H*W,C))
  #   x_reshaped=np.reshape(x[:,i,:,:],(N*H*W,1))
  #   out_before_reshape, cache=batchnorm_forward(x_reshaped, gamma, beta, bn_param)
  #   print "out_before_reshape:",out_before_reshape.shape
  #   out[:,i,:,:]=np.reshape(out_before_reshape,(N, H, W))
     
  N, C, H, W = x.shape
  x_reshaped=x.transpose(0,2,3,1).reshape(N*H*W,C)
  out,cache=batchnorm_forward(x_reshaped,gamma,beta,bn_param)
  out=out.reshape(N,H,W,C).transpose(0,3,1,2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache
  
  
def spatial_batchnorm_forward_raw(x, gamma, beta, bn_param): #an implementation can work by looping though each channel.
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  #batchnorm_forward(x, gamma, beta, bn_param)
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, C, H, W= x.shape
  running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

  sample_mean=np.mean(x,axis=(0,2,3))
  sample_var=np.std(x,axis=(0,2,3))
  X_normalized=np.zeros_like(x)
  out=np.zeros_like(x)
  if mode == 'train':
      for i in xrange(C):
          X_normalized[:,i,:,:]=(x[:,i,:,:]-sample_mean[i])/np.sqrt(sample_var[i]**2+eps)
          out[:,i,:,:]=gamma[i]*X_normalized[:,i,:,:]+beta[i]
          running_mean[i]=momentum*running_mean[i]+(1-momentum)*sample_mean[i]
          running_var[i]=momentum*running_var[i]+(1-momentum)*sample_var[i]
      running_mean=momentum*running_mean+(1-momentum)*sample_mean
      running_var=momentum*running_var+(1-momentum)*sample_var
      bn_param['running_mean'] = running_mean
      bn_param['running_var'] = running_var
  elif mode == 'test':
      for i in xrange(C):
          X_normalized[:,i,:,:]=(x[:,i,:,:]-running_mean[i])/running_var[i]
          out[:,i,:,:]=gamma[i]*X_normalized[:,i,:,:]+beta[i]
  #save intermediates to cache
  cache=(x,sample_mean,sample_var,gamma,beta,eps,X_normalized)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N,C,H,W=dout.shape
  #1.reshape orginal data
  dout=dout.transpose(0,2,3,1).reshape(N*H*W,C)
  #2.call vanilla version of batch normalization, to get result
  dx_2d, dgamma, dbeta=batchnorm_backward(dout, cache)
  #3.reshape result to meet require result
  dx=dx_2d.reshape(N,H,W,C).transpose(0,3,1,2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
