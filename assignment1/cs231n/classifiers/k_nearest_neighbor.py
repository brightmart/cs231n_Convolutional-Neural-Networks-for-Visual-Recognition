import numpy as np
from scipy import stats
class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
	#print num_test
    num_train = self.X_train.shape[0]
	#print num_train
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):


        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
		  singleDist=X[i,:]-self.X_train[j,:]
		  #print("singleDist",singleDist)
		  squareDist=np.square(singleDist)
		  #print("squareDist",squareDist)
		  sum=np.sum(squareDist)
		  #print("sum",sum)
		  dists[i,j]=np.sqrt(sum)
		  #print("dists[i,j]",dists[i,j])
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
        
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      ####################################################################### 
         dist=X[i,:]-self.X_train
         #print("dist.shape:",dist.shape)
         squared_dist=np.square(dist)
         #print("squared_dist.shape:",squared_dist.shape)
         sum_squared_dist=np.sum(squared_dist,axis=1)
         #print("sum_squared_dist.shape:",sum_squared_dist.shape)
         i_dists=np.sqrt(sum_squared_dist)
         #print("i_dists.shape",i_dists.shape,"dists[i,:]:",dists[i,:].shape) #('i_dists.shape', (5000L, 32L, 3L), 'dists[i,:]:', (5000L,))
         
         dists[i,:]=i_dists
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    #dists1=(np.square(X)).dot(np.ones((X.shape[1],num_train)))-2*X.dot(self.X_train.T)+np.ones((num_test,X.shape[1])).dot(np.square(self.X_train).T) 
    dists1_X_square=(np.square(X)).dot(np.ones((X.shape[1],num_train)))
    dists1_X_dot_Xtrain=-2.0*X.dot(self.X_train.T)
    dists1_Xtrain_square=np.ones((num_test,X.shape[1])).dot(np.square(self.X_train).T) 
    dists2=dists1_X_square+dists1_X_dot_Xtrain+dists1_Xtrain_square
    dists=np.sqrt(dists2)
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    #print "k:",k
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
        closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
        dists2ith=dists[i,:]
        neighbors=np.argsort(dists2ith)
        nearest_neighbors_k=neighbors[:k]
        closest_y=self.y_train[nearest_neighbors_k]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
        label=stats.mode(closest_y)[0] #get mode from array using scipy.stats.mode
        y_pred[i]=label
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

