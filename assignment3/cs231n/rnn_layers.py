import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  #pass
  sum=prev_h.dot(Wh)+x.dot(Wx)+b
  next_h=np.tanh(sum)
  cache=(x,prev_h,Wx,Wh,b,next_h)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state #(N,H)
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  (x,prev_h,Wx,Wh,b,next_h)=cache
  gradient_parent=dnext_h*(1.0-(next_h*next_h))
  
  db=gradient_parent.T.dot(np.ones((gradient_parent.shape[0]))) #db=gradient_parent.sum(axis=0)
  dx=gradient_parent.dot(Wx.T)
  dprev_h=gradient_parent.dot(Wh.T)
  dWx=(x.T).dot(gradient_parent)
  dWh=prev_h.T.dot(gradient_parent)
  
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  #pass
  N,T,D=x.shape
  N,H=h0.shape
  h=np.zeros((N,T,H))
  cache={}
  x_=x.transpose(1,0,2) #transfer x to (T,N,D)
  prev_h=h0
  
  for i in xrange(T):
      next_h, _=rnn_step_forward(x_[i], prev_h, Wx, Wh, b)  #_ is cache,no use.
      h[:,i,:]=next_h
      cache[i]=(x_[i],prev_h,Wx,Wh,b,next_h) #save cache for use of rnn_backward
      prev_h=next_h # update prev_h, for next iteration.
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache

def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data. 
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H) 
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  N,T,H=dh.shape
  _,D=cache[0][0].shape
  dh_=dh.transpose(1,0,2) #transfer dh to (T,N,D)
  dx_=np.zeros((T,N,D))
  dh0=np.zeros((N,H))
  dprev_h=np.zeros((N,H))
  dWx=np.zeros((D,H))
  dWh=np.zeros((H,H))
  db=np.zeros((H,)) 
  i=T-1
  while i>=0:
      dnext_h=dh_[i] 
      #dprev_h: Gradients of previous hidden state, of shape (N, H); dnext_h---> Gradient of loss with respect to next hidden state. dnext_h:(N,H)
      dx_[i], dprev_h, dWx_, dWh_, db_=rnn_step_backward(dnext_h+dprev_h, cache[i]) #IMPORTANT rnn_step_backward(
      dWx+=dWx_
      dWh+=dWh_
      db+=db_
      i=i-1
  dh0=dprev_h
  dx=dx_.transpose(1,0,2)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  #pass
  N,T=x.shape
  V,D=W.shape
  out=np.zeros((N,T,D))
  for i in xrange(N):
      out[i,:,:]=W[x[i,:]]
  cache=(x,W)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  #pass
  N,T,D=dout.shape
  x,W=cache
  dW=np.zeros(W.shape)
  
  for i in xrange(N):
      indices=x[i,:]
      np.add.at(dW,indices,dout[i,:,:])
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward_MINE(x, prev_h, prev_c, Wx, Wh, b): #mine. works. alright.
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  #pass
  N,H=prev_h.shape
  #1.compute an activation vector a. demension is (N,4H)
  a=x.dot(Wx)+prev_h.dot(Wh)+b 
  #2.1 divide a into four vectors: ai,af,ao,ag. each demension is (N,H)
  ai=a[:,0:H] 
  af=a[:,H:2*H]
  ao=a[:,2*H:3*H]
  ag=a[:,3*H:4*H]
  #2.2 compute i,f,o,g using sigmoid or tanh.
  i=sigmoid(ai) 
  f=sigmoid(af)
  o=sigmoid(ao)
  g=np.tanh(ag)
  #3.1 compuate next cell state
  next_c=f*prev_c+i*g 
  #next_h=o*np.tanh(next_c) #3.2 compute next hidden state
  next_c_tanh=np.tanh(next_c)
  next_h=o*next_c_tanh
  #save cache
  #cache=x, prev_h, prev_c, Wx, Wh, b,a,ai,af,ao,ag,i,f,o,g,next_c,next_h
  cache=next_h,next_c, Wx, Wh, b,prev_h, prev_c,x
  #cache=x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_c
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ############################################################################## 
  return next_h, next_c, cache


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):#other
    """
    Forward pass for a single timestep of an LSTM.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    _, H = prev_c.shape
    ##########################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    ##########################################################################
    score = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    i, f, o, g = score[:, :H], score[
        :, H:2 * H], score[:, 2 * H:3 * H], score[:, 3 * H:4 * H]
    i, f, o, g = sigmoid(i), sigmoid(f), sigmoid(o), np.tanh(g)
    next_c = np.multiply(f, prev_c) + np.multiply(i, g)
    next_h = np.multiply(o, np.tanh(next_c))
    ##########################################################################
    #                               END OF YOUR CODE                             #
    ##########################################################################
    cache = (x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_c)
    return next_h, next_c, cache

def lstm_step_backward(dnext_h, dnext_c, cache):#other
    """
    Backward pass for a single timestep of an LSTM.
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    dprev_h = None
    ##########################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    ##########################################################################

    x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_c = cache
    tanh_nextc = np.tanh(next_c)
    one_minus_tanh_nextc_square = 1 - np.power(tanh_nextc, 2)
    dprev_c = np.multiply(dnext_c, f)
    dprev_c += np.multiply(np.multiply(o,
                                       np.multiply(one_minus_tanh_nextc_square, f)), dnext_h)

    d_i = np.multiply(dnext_c, g)
    d_i += np.multiply(np.multiply(o,
                                   np.multiply(one_minus_tanh_nextc_square, g)), dnext_h)
    d_f = np.multiply(dnext_c, prev_c)
    d_f += np.multiply(np.multiply(o,
                                   np.multiply(one_minus_tanh_nextc_square, prev_c)), dnext_h)
    d_o = np.multiply(dnext_h, tanh_nextc)
    d_g = np.multiply(dnext_c, i)
    d_g += np.multiply(np.multiply(o,
                                   np.multiply(one_minus_tanh_nextc_square, i)), dnext_h)

    d_i = np.multiply(d_i, i * (1 - i))
    d_f = np.multiply(d_f, f * (1 - f))
    d_o = np.multiply(d_o, o * (1 - o))
    d_g = np.multiply(d_g, 1 - np.power(g, 2))

    difog = np.hstack((d_i, d_f, d_o, d_g))

    dWh = np.dot(prev_h.T, difog)
    dWx = np.dot(x.T, difog)
    db = difog.sum(axis=0)

    dx = np.dot(difog, Wx.T)
    dprev_h = np.dot(difog, Wh.T)
    ##########################################################################
    #                               END OF YOUR CODE                             #
    ##########################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_step_backward_MINE(dnext_h, dnext_c, cache):#mine.works. alright.
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  #x, prev_h, prev_c, Wx, Wh, b,a,ai,af,ao,ag,i,f,o,g,next_c,next_h=cache
  next_h,next_c, Wx, Wh, b,prev_h, prev_c,x=cache
  #next_h,next_c, Wx, Wh, b,prev_h, prev_c,x=cache
  N,H=prev_h.shape
  #step1: compute do,dc
  tanh_nextc = np.tanh(next_c)
  one_minus_tanh_nextc_square=1-np.power(tanh_nextc,2)
  #dtanh_c=dnext_h*o*(1-np.tanh(next_c)**2) #need remove  
  #step2:compute di,df,da,dprev_c
  ##################################################################################################################
  #additional step to get intermediate parameters
  a=x.dot(Wx)+prev_h.dot(Wh)+b 
  #2.1 divide a into four vectors: ai,af,ao,ag. each demension is (N,H)
  ai=a[:,0:H] 
  af=a[:,H:2*H]
  ao=a[:,2*H:3*H]
  ag=a[:,3*H:4*H]
  #2.2 compute i,f,o,g using sigmoid or tanh.
  i=sigmoid(ai) 
  f=sigmoid(af)
  o=sigmoid(ao)
  g=np.tanh(ag)
  ###################################################################################################################
  dprev_c=np.multiply(dnext_c,f)
  dprev_c += np.multiply(np.multiply(o, #QUESTION 1?
                                       np.multiply(one_minus_tanh_nextc_square, f)), dnext_h)
  di=np.multiply(dnext_c,g)
  di += np.multiply(np.multiply(o, #QUESTION 2?
                                   np.multiply(one_minus_tanh_nextc_square, g)), dnext_h)
  df=np.multiply(dnext_c,prev_c)
  df += np.multiply(np.multiply(o, #QUESTION 3?
                                   np.multiply(one_minus_tanh_nextc_square, prev_c)), dnext_h)
  do=np.multiply(dnext_h,tanh_nextc)
  dg=np.multiply(dnext_c,i)
  dg += np.multiply(np.multiply(o, #QUESTION 4?
                                   np.multiply(one_minus_tanh_nextc_square, i)), dnext_h)

  #step3: compute da0,dai,dag,daf
  dai=di*(i*(1-i)) #equals to np.multiply(di,(i*(1-i))
  daf=df*(f*(1-f))
  dao=do*(o*(1-o))
  dag=dg*(1-g**2)
  #step4: compute da by accumulate 4 parts
  da=np.zeros_like(a)
  da[:,0:H]=dai
  da[:,H:2*H]=daf
  da[:,2*H:3*H]=dao
  da[:,3*H:4*H]=dag
  #print "da:",da
  #ddaa=[dai,daf,dao,dag]
  #print "ddaa.shape",ddaa.shape
  #step5: final step. gradient for x,Wx,prev_h,wh,b
  db=np.sum(da,axis=0)
  dx=da.dot(Wx.T) # a=x.dot(Wx)+prev_h.dot(Wh)+b 
  dWx=x.T.dot(da)
  dprev_h=da.dot(Wh.T)
  dWh=prev_h.T.dot(da)
  #
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dprev_c, dWx, dWh, db

def lstm_forward_classic(x, h0, Wx, Wh, b):
   h, cache = None, None
   _, H = h0.shape
   N, T, D = x.shape
   h = np.zeros((N, T, H)) 

   prev_h = h0
   prev_c = np.zeros(h0.shape)
   cache = []
   for i in range(T):
       th, tc, tcache = lstm_step_forward(
       x[:, i, :], prev_h, prev_c, Wx, Wh, b)
       prev_h = th
       prev_c = tc
       h[:, i, :] = th
   cache.append(tcache)
   return h, cache

def lstm_forward_MINE(x, h0, Wx, Wh, b): #works. alright
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N,T,D=x.shape
  _,H=h0.shape
  #1.transfer x fom (N,T,D) to (T,N,D). Single step x-----> x: Input data, of shape (N, D)
  #x_=x.transpose(1,0,2)
  h=np.zeros((N, T, H))
  prev_h=h0
  prev_c=np.zeros((N,H))
  cache={}
  #2.using lstm_step_foward to loop each timestamp t.
  for i in xrange(T): # prev_h: Previous hidden state, of shape (N, H). prev_c: previous cell state, of shape (N, H)
      x_i=x[:,i,:]
      next_h, next_c, _=lstm_step_forward(x_i, prev_h, prev_c, Wx, Wh, b) # 3.make one single step x_[i,:,:]
      h[:,i,:]=next_h # 4.assign value to h
      prev_h=next_h
      prev_c=next_c# 5.update hidden state,cell state
      cache[i]=next_h,next_c, Wx, Wh, b,prev_h, prev_c,x_i#save cache for backward of lstm.
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache

def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.
    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.
    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    _, H = h0.shape
    N, T, D = x.shape
    h = np.zeros((N, T, H))
    ############################################################

    ###########################
    ##########################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    ##########################################################################
    prev_h = h0
    prev_c = np.zeros(h0.shape)
    cache = []
    for i in range(T):
        th, tc, tcache = lstm_step_forward(
            x[:, i, :], prev_h, prev_c, Wx, Wh, b)
        prev_h = th
        prev_c = tc
        h[:, i, :] = th
        cache.append(tcache)
    ##########################################################################
    #                               END OF YOUR CODE                             #
    ##########################################################################

    return h, cache


def lstm_backward(dh, cache):#other
    """
    Backward pass for an LSTM over an entire sequence of data.]
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    ##########################################################################
    N, T, H = dh.shape
    D = cache[0][0].shape[1]
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4 * H))
    dWh = np.zeros((H, 4 * H))
    db = np.zeros((4 * H))
    tdprev_h = np.zeros((N, H))
    tdprev_c = np.zeros((N, H))
    #######################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above.                                                             #
    #######################################################################
    for i in range(T - 1, -1, -1):
        tdx, tdprev_h, tdprev_c, tdWx, tdWh, tdb = lstm_step_backward(dh[:, i, :] + tdprev_h,
                                                                      tdprev_c, cache[i])
        dx[:, i, :] = tdx
        dWx += tdWx
        dWh += tdWh
        db += tdb
    dh0 = tdprev_h

    ##########################################################################
    #                               END OF YOUR CODE                             #
    ##########################################################################

    return dx, dh0, dWx, dWh, db


def lstm_backward_MINE(dh, cache): #mine: almost here, but why error difference is 1.0
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  #loop from end to begining using lstm_step_backward. try to add additionally gradient for hidden state and cell state, if possible.
  N,T,H=dh.shape
  next_h,next_c, Wx, Wh, b,prev_h, prev_c,x= cache[0] #init dx,dh0,dWx,dWh,db  --->next_h,next_c, Wx, Wh, b,prev_h, prev_c,x_
  _,D=x.shape
  dx=np.zeros((N,T,D))
  dh0=np.zeros_like(prev_h)
  dWx=np.zeros_like(Wx)
  dWh=np.zeros_like(Wh)
  db=np.zeros_like(b)
  i=T-1
  dprev_h=np.zeros_like(prev_h)
  dprev_c=np.zeros_like(next_c)
  while i>=0: #loop T
      #dnext_h=dh[:,i,:]+dprev_h #important..........
      dx_, dprev_h, dprev_c, dWx_, dWh_, db_=lstm_step_backward(dh[:, i, :] + dprev_h, dprev_c, cache[i]) #2.one step backward of LSTM .dprev_h
      dx[:,i,:]=dx_  #3.accumulate gradient of x,Wx,Wh,b
      dWx+=dWx_
      dWh+=dWh_
      db+=db_  
      i=i-1 #4.move index
  dh0=dprev_h     
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def lstm_backward_classic4(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    ##########################################################################

    # Backprop into the lstm.

    # Dimensions
    (N, T, H) = dh.shape
    next_h,next_c, Wx, Wh, b,prev_h, prev_c,x= cache[0] 
    _,D=x.shape

    # Initialize dx,dh0,dWx,dWh,db
    dx = np.zeros((T, N, D))
    dh0 = np.zeros((N, H))
    db = np.zeros((4 * H,))
    dWh = np.zeros((H, 4 * H))
    dWx = np.zeros((D, 4 * H))

    # On transpose dh
    dh = dh.transpose(1, 0, 2)
    dh_prev = np.zeros((N, H))
    dc_prev = np.zeros_like(dh_prev)

    for t in reversed(xrange(T)):
        dh_current = dh[t] + dh_prev
        dc_current = dc_prev
        dx_t, dh_prev, dc_prev, dWx_t, dWh_t, db_t = lstm_step_backward(
            dh_current, dc_current, cache[t])
        dx[t] += dx_t
        dh0 = dh_prev
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    dx = dx.transpose(1, 0, 2)
    return dx, dh0, dWx, dWh, db

def lstm_backward_classic3(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  #N, T, D, H, cache_t = cache
  (N, T, H) = dh.shape
  next_h,next_c, Wx, Wh, b,prev_h, prev_c,x= cache[0] 
  _,D=x.shape
  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, 4 * H))
  dWh = np.zeros((H, 4 * H))
  db = np.zeros((4 * H,))

  dnext_c = np.zeros((N, H))
  dnext_h = dh[:, T - 1, :]
  for t in reversed(range(T)):
    lstm_cache = cache[t]
    dx_t, dprev_h_t, dprev_c_t, dWx_t, dWh_t, db_t = lstm_step_backward(dnext_h, dnext_c, lstm_cache)

    if t > 0:
      dnext_h = dprev_h_t + dh[:, t - 1, :]
      dnext_c = dprev_c_t

    dx[:, t, :] += dx_t
    dWx += dWx_t
    dWh += dWh_t
    db += db_t

  dh0 = dprev_h_t
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db

def lstm_backward_classic2(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  (N, T, H) = dh.shape
  next_h,next_c, Wx, Wh, b,prev_h, prev_c,x= cache[0] 
  #x,prev_h, prev_c, Wx, Wh, i,f,o,g, next_h, next_c= cache[T-1]

  N,D = x.shape
  dx = np.zeros((N,T,D))
  dWx = np.zeros(Wx.shape)
  dWh = np.zeros(Wh.shape)
  db = np.zeros((4*H))
  dprev = np.zeros(prev_h.shape)
  dprev_c = np.zeros(prev_c.shape)

  for t in range(T-1,-1,-1):
      dx[:,t,:], dprev, dprev_c, dWx_local, dWh_local, db_local = lstm_step_backward(dh[:,t,:]+dprev, dprev_c, cache[t])
      dWx+=dWx_local
      dWh+=dWh_local
      db +=db_local

  dh0 = dprev 
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def lstm_backward_classic1(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    ##########################################################################
    N, T, H = dh.shape
    #D = cache[0][0].shape[1]
    next_h,next_c, Wx, Wh, b,prev_h, prev_c,x= cache[0] #init dx,dh0,dWx,dWh,db
    _,D=x.shape
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4 * H))
    dWh = np.zeros((H, 4 * H))
    db = np.zeros((4 * H))
    tdprev_h = np.zeros((N, H))
    tdprev_c = np.zeros((N, H))
    #######################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above.                                                             #
    #######################################################################
    for i in range(T - 1, -1, -1):
        print "i:",i
        tdx, tdprev_h, tdprev_c, tdWx, tdWh, tdb = lstm_step_backward(dh[:, i, :] + tdprev_h,
                                                                      tdprev_c, cache[i])

        dx[:, i, :] = tdx
        dWx += tdWx
        dWh += tdWh
        db += tdb
    dh0 = tdprev_h

    ##########################################################################
    #                               END OF YOUR CODE                             #
    ##########################################################################

    return dx, dh0, dWx, dWh, db

def temporal_affine_forward(x, w, b): #official implementation of temporal_affine_forward
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache
  
def temporal_affine_backward_brightversion(dout,cache):#CAN WORK.
    N,T,M=dout.shape
    x,w,b,out=cache
    _,_,D=x.shape
    db=dout.reshape(N*T,M).T.dot(np.ones((N*T)))
    dx=dout.reshape(N*T,M).dot(w.T).reshape(N,T,D)
    dw=(x.transpose(2,0,1).reshape(D,N*T)).dot(dout.reshape(N*T,M))
    return dx,dw,db
    
def temporal_affine_forward_brightversion(x,w,b): #imple of temporal_affine_forward with loop( brightmart's simple version)
    N,T,D=x.shape
    _,M=w.shape
    x_=x.transpose(1,0,2) #transfer x  to T,N,D
    out=np.zeros((T,N,M))
    for i in xrange(T):
        out[i,:,:]=x_[i,:,:].dot(w)+b
    out=out.transpose(1,0,2)
    cache=x,w,b,out
    return out,cache
    
def temporal_affine_backward(dout, cache):#official implemenation of temporal_affine_backward
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

