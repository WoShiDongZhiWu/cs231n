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
  pass
  num_train = X.shape[0]
  num_class = W.shape[1]
# Li = -fyi + log(sum(exp(fj))) 
# Li = -WyiXi + log(sum(exp(WjXi))) 
# 求导数的时候需要注意，第二项中包含正确的类别的情况,即Wj包含Wyi
# 对Wyi求导，得dLi = (-1 + exp(WyiX)/sum(exp(WjX)))Xi；对Wj求导,得exp(WjXi)/sum(exp(WjXi))Xi
  for i in np.arange(num_train):
        scores = X[i].dot(W) # 1 × C
        scores -= np.max(scores) #防止数值太大导致计算不稳定数值爆炸，平移数值使最大值为0
        exp_scores = np.exp(scores)
        scores_correct_class = scores[y[i]]
        scores_j = 0.0  
        dW[:,y[i]] -= X[i].T
        for j in np.arange(num_class):
            scores_j += exp_scores[j]
            dW[:,j] += exp_scores[j]/np.sum(exp_scores)* X[i].T
        loss += -scores_correct_class + np.log(scores_j)
        
  loss /= num_train #得到均值
  loss += reg*np.sum(W*W) #L2正则化
  dW /= num_train
  dW += reg*W
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
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W) # N×C
  scores -= np.max(scores,axis=1,keepdims=True) #在列方向上平移
  exp_scores = np.exp(scores)
  sum_row_exp_scores = np.sum(exp_scores,axis = 1).reshape(1,-1) #1×N
  correct_scores = exp_scores[np.arange(X.shape[0]),y] #每类正确的标签对应的得分
  loss -= np.sum(np.log(correct_scores/sum_row_exp_scores))
  loss /= num_train
  loss += reg*np.sum(W*W) #L2正则化
  # 对Wyi求导，得dLi = (-1 + exp(WyiX)/sum(exp(WjX)))Xi；对Wj求导,得exp(WjXi)/sum(exp(WjXi))Xi
  p = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)
  #y_lable就是(N,C)维的矩阵，每一行中只有对应的那个正确类别 = 1，其他都是0
  y_lable = np.zeros((num_train,num_class))
  y_lable[np.arange(num_train),y] = 1
  dW = X.T.dot(p-y_lable)
  dW /= num_train
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

