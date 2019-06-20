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
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i].T
        dW[:,y[i]] -= X[i].T
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


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
  pass
  scores = X.dot(W) # N行image对应的C列class的评分
  # 第一行样本对应的正确类别是y[0]，即第一行对应的正确类别的得分在第一行第y[0]列
  correct_scores = np.array(scores[np.arange(X.shape[0]),y])  # 数组为1×N
  correct_scores = np.reshape(correct_scores,(-1,1)) # 将数组转化为N×1（每行data对应的正确类别得分）
  margin = scores - correct_scores +1 # delta = 1    
  # 这时对角线处的值为1，应该设置为0（根据svm loss公式）
  margin[np.arange(X.shape[0]),y] = 0.0
  loss_matrix = np.maximum(margin, 0) #  max（0，margin），得到损失函数矩阵
  loss = np.sum(loss_matrix)/X.shape[0] + reg*np.sum(W*W)  # sum求和，求均值并添加正则项；得到loss值
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  # 借用求loss时得到的中间量margin N×C，对角线为0
  # 大于0的值设置为1
  margin[margin>0] = 1
  # 不大于0的值设置为0
  margin[margin<0] = 0
  # X N×D
  row_sum = np.sum(margin,axis=1)
  margin[np.arange(X.shape[0]),y] = -row_sum #根据公式，每行样本的正确类别对应的计算
  dW = X.T.dot(margin)
  dW /= X.shape[0] #求每个样本的均值
  dW += reg* W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
