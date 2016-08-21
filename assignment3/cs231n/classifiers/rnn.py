import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CaptioningRNN(object):
  """
  A CaptioningRNN produces captions from image features using a recurrent
  neural network.

  The RNN receives input vectors of size D, has a vocab size of V, works on
  sequences of length T, has an RNN hidden dimension of H, uses word vectors
  of dimension W, and operates on minibatches of size N.

  Note that we don't use any regularization for the CaptioningRNN.
  """
  
  def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
               hidden_dim=128, cell_type='rnn', dtype=np.float32):
    """
    Construct a new CaptioningRNN instance.

    Inputs:
    - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
      and maps each string to a unique integer in the range [0, V).
    - input_dim: Dimension D of input image feature vectors.
    - wordvec_dim: Dimension W of word vectors.
    - hidden_dim: Dimension H for the hidden state of the RNN.
    - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
    - dtype: numpy datatype to use; use float32 for training and float64 for
      numeric gradient checking.
    """
    if cell_type not in {'rnn', 'lstm'}:
      raise ValueError('Invalid cell_type "%s"' % cell_type)
    
    self.cell_type = cell_type
    self.dtype = dtype
    self.word_to_idx = word_to_idx
    self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
    self.params = {}
    
    vocab_size = len(word_to_idx)

    self._null = word_to_idx['<NULL>']
    self._start = word_to_idx.get('<START>', None)
    self._end = word_to_idx.get('<END>', None)
    
    # Initialize word vectors
    self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
    self.params['W_embed'] /= 100
    
    # Initialize CNN -> hidden state projection parameters
    self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
    self.params['W_proj'] /= np.sqrt(input_dim)
    self.params['b_proj'] = np.zeros(hidden_dim)

    # Initialize parameters for the RNN
    dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
    self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
    self.params['Wx'] /= np.sqrt(wordvec_dim)
    self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
    self.params['Wh'] /= np.sqrt(hidden_dim)
    self.params['b'] = np.zeros(dim_mul * hidden_dim)
    
    # Initialize output to vocab weights
    self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
    self.params['W_vocab'] /= np.sqrt(hidden_dim)
    self.params['b_vocab'] = np.zeros(vocab_size)
      
    # Cast parameters to correct dtype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(self.dtype)


  def loss(self, features, captions):
    """
    Compute training-time loss for the RNN. We input image features and
    ground-truth captions for those images, and use an RNN (or LSTM) to compute
    loss and gradients on all parameters.
    
    Inputs:
    - features: Input image features, of shape (N, D)
    - captions: Ground-truth captions; an integer array of shape (N, T) where
      each element is in the range 0 <= y[i, t] < V
      
    Returns a tuple of:
    - loss: Scalar loss
    - grads: Dictionary of gradients parallel to self.params
    """
    # Cut captions into two pieces: captions_in has everything but the last word
    # and will be input to the RNN; captions_out has everything but the first
    # word and this is what we will expect the RNN to generate. These are offset
    # by one relative to each other because the RNN should produce word (t+1)
    # after receiving word t. The first element of captions_in will be the START
    # token, and the first element of captions_out will be the first word.
    captions_in = captions[:, :-1]
    captions_out = captions[:, 1:]
    
    # You'll need this 
    mask = (captions_out != self._null)

    # Weight and bias for the affine transform from image features to initial
    # hidden state
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    
    # Word embedding matrix
    W_embed = self.params['W_embed']

    # Input-to-hidden, hidden-to-hidden, and biases for the RNN
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

    # Weight and bias for the hidden-to-vocab transformation.
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    
    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
    # In the forward pass you will need to do the following:                   #
    # (1) Use an affine transformation to compute the initial hidden state     #
    #     from the image features. This should produce an array of shape (N, H)#
    # (2) Use a word embedding layer to transform the words in captions_in     #
    #     from indices to vectors, giving an array of shape (N, T, W).         #
    # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
    #     process the sequence of input word vectors and produce hidden state  #
    #     vectors for all timesteps, producing an array of shape (N, T, H).    #
    # (4) Use a (temporal) affine transformation to compute scores over the    #
    #     vocabulary at every timestep using the hidden states, giving an      #
    #     array of shape (N, T, V).                                            #
    # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
    #     the points where the output word is <NULL> using the mask above.     #
    #                                                                          #
    # In the backward pass you will need to compute the gradient of the loss   #
    # with respect to all model parameters. Use the loss and grads variables   #
    # defined above to store loss and gradients; grads[k] should give the      #
    # gradients for self.params[k].                                            #
    ############################################################################
    #forward pass
    #1 to compute the initial hidden state. (N, H)
    out_features=features.dot(W_proj)+b_proj 
    #2 transform the words in captions_in from indices to vectors.(N, T, W)
    out_wordvector, cache_wordvector = word_embedding_forward(captions_in, W_embed)
    #3 process the sequence of input word vectors and produce hidden state vectors for all timesteps, producing an array of shape (N, T, H).
    if self.cell_type=='rnn':
        hidden_state, cache_hiddenstate = rnn_forward(out_wordvector, out_features, Wx, Wh, b) # 
    elif self.cell_type=='lstm': #add for LSTM.
        hidden_state, cache_hiddenstate = lstm_forward(out_wordvector, out_features, Wx, Wh, b) #x, h0, Wx, Wh, b
    #4 compute scores over the vocabulary at every timestep using the hidden states, giving an array of shape (N, T, V). 
    out_scores, cache_scores = temporal_affine_forward(hidden_state, W_vocab, b_vocab) 
    #5 compute loss using captions_out
    loss, dout=temporal_softmax_loss(out_scores, captions_out, mask)
    
    #backward pass
    dhidden_state, grads['W_vocab'], grads['b_vocab']=temporal_affine_backward(dout, cache_scores) #backward 1
    if self.cell_type=='rnn':
        dout_wordvector, dout_features, grads['Wx'], grads['Wh'], grads['b']=rnn_backward(dhidden_state, cache_hiddenstate) #backward 2 for rnn 
    elif self.cell_type=='lstm': #add for LSTM.
        dout_wordvector, dout_features, grads['Wx'], grads['Wh'], grads['b']=lstm_backward(dhidden_state, cache_hiddenstate) #dh, cache---->dx, dh0, dWx, dWh, db
    grads['W_embed']=word_embedding_backward(dout_wordvector, cache_wordvector) #backward 3
    grads['b_proj']=np.sum(dout_features,axis=0)
    grads['W_proj']=features.T.dot(dout_features)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads


  def sample_mine(self, features, max_length=30):
    """
    Run a test-time forward pass for the model, sampling captions for input
    feature vectors.

    At each timestep, we embed the current word, pass it and the previous hidden
    state to the RNN to get the next hidden state, use the hidden state to get
    scores for all vocab words, and choose the word with the highest score as
    the next word. The initial hidden state is computed by applying an affine
    transform to the input image features, and the initial word is the <START>
    token.

    For LSTMs you will also have to keep track of the cell state; in that case
    the initial cell state should be zero.

    Inputs:
    - features: Array of input image features of shape (N, D).
    - max_length: Maximum length T of generated captions.

    Returns:
    - captions: Array of shape (N, max_length) giving sampled captions,
      where each element is an integer in the range [0, V). The first element
      of captions should be the first sampled word, not the <START> token.
    """
    N = features.shape[0]
    captions = self._null * np.ones((N, max_length), dtype=np.int32)

    # Unpack parameters
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    W_embed = self.params['W_embed']
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    
    ###########################################################################
    # TODO: Implement test-time sampling for the model. You will need to      #
    # initialize the hidden state of the RNN by applying the learned affine   #
    # transform to the input image features. The first word that you feed to  #
    # the RNN should be the <START> token; its value is stored in the         #
    # variable self._start. At each timestep you will need to do to:          #
    # (1) Embed the previous word using the learned word embeddings           #
    # (2) Make an RNN step using the previous hidden state and the embedded   #
    #     current word to get the next hidden state.                          #
    # (3) Apply the learned affine transformation to the next hidden state to #
    #     get scores for all words in the vocabulary                          #
    # (4) Select the word with the highest score as the next word, writing it #
    #     to the appropriate slot in the captions variable                    #
    #                                                                         #
    # For simplicity, you do not need to stop generating after an <END> token #
    # is sampled, but you can if you want to.                                 #
    #                                                                         #
    # HINT: You will not be able to use the rnn_forward or lstm_forward       #
    # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
    # a loop.                                                                 #
    ###########################################################################
    #1 Embed the previous word using the learned word embeddings
    caption=self._start
    for i in xrange(max_length):
        #print "i:",i
        #1. Embed the previous word using the learned word embeddings
        caption = np.ones((N, 1), dtype=np.int32) * self._start 
        vector, _ = word_embedding_forward(caption, W_embed) # 
        prev_h=features.dot(self.params['W_proj'])+self.params['b_proj'] 
        #2. Make an RNN step using the previous hidden state and the embedded current word to get the next hidden state.
        #print "vector:",vector
        next_h, _=rnn_step_forward(vector, prev_h, self.params['Wx'], self.params['Wh'], self.params['b']) #x, prev_h, Wx, Wh, b 
        #3. Apply the learned affine transformation to the next hidden state to get scores for all words in the vocabulary.
        out_scores, _ = next_h.dot(self.params['W_vocab'])+self.params['b_vocab'] #features.dot(W_proj)+b_proj 
        #4. Select the word with the highest score as the next word , writing it to the appropriate slot in the captions variable
        prev_h=next_h
        print "out_scores.shape:",out_scores.shape
        index_max_value=np.argmax(out_scores,axis=1)
        captions[:,i]=index_max_value
        caption=index_max_value
        #print "caption:",caption
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return captions

  def sample(self, features, max_length=30): #right....
      print "start..."
      N = features.shape[0]
      captions = self._null * np.ones((N, max_length), dtype=np.int32)
      W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
      W_embed = self.params['W_embed']
      Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
      W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
      
      h0 = features 
      _,H=h0.shape
      h0, h0_cache = affine_forward(h0, W_proj, b_proj)  #
      word = np.ones((N, 1), dtype=np.int32) * self._start
      prev_c=np.zeros((N,H))
      for i in range(max_length): 
          word=word.reshape(N,1)
          #1. Embed the previous word using the learned word embeddings
          wembed, wembed_cache = word_embedding_forward(word, W_embed)  
          # rnn_h, rnn_h_cache = rnn_forward(wembed, h0, Wx, Wh, b) #actually we can use rnn_forward 
          #2. Make an RNN step using the previous hidden state and the embedded current word to get the next hidden state.
          if self.cell_type=='rnn':
              next_h, _ = rnn_step_forward(np.squeeze(wembed), h0, Wx, Wh, b) #question1: why need squeeze?
          elif self.cell_type=='lstm':
              next_h, prev_c, _=lstm_step_forward(np.squeeze(wembed), h0, prev_c, Wx, Wh, b) #x, prev_h, prev_c, Wx, Wh, b
          #3. Apply the learned affine transformation to the next hidden state to get scores for all words in the vocabulary.
          rnn_out_score, _ = temporal_affine_forward(next_h.reshape((N,1,-1)), W_vocab, b_vocab) 
          h0 = next_h 
          #4. Select the word with the highest score as the next word , writing it to the appropriate slot in the captions variable
          #print "out_scores.shape:",rnn_out_score.shape #(3L, 1L, 1004L)--->(3L, 1004L)--->3
          predicted_word = np.argmax(np.squeeze(rnn_out_score), axis=1) #question2: axis=1---> sum according to column,get records number sames as rows.
          word = predicted_word 
          captions[:, i] = word 
      print "end..."
         ####################################################################### 
         #                             END OF YOUR CODE                             # 
         ####################################################################### 
      return captions 
