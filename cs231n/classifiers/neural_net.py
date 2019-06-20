from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  一个两层的全连接网络（一个隐藏层）。 输入维度N, 一个隐藏层的维度 H, 对 C 个类别进行分类。
  利用softmax损失函数和L2正则化训练网络的权重矩阵。
  在第一个全连接层后，使用一个ReLU非线性函数。

  换句话说，该网络的结构如下：
  输入 - 全连接层 - ReLU - 全连接层 - softmax
  第二个全连接层的输出为每个类别的得分。
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    初始化模型. Weights初始化为小的随机数，biases初始化为0。
    Weights and biases存在变量 self.params中, 该变量是一个字典，并有如下的keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size) # (D, H)
    self.params['b1'] = np.zeros(hidden_size) # (H,)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size) # (H, C)
    self.params['b2'] = np.zeros(output_size) # (C,)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # 从字典参数中读取变量
    W1, b1 = self.params['W1'], self.params['b1'] # (D, H) ,(H,)
    W2, b2 = self.params['W2'], self.params['b2'] # (H, C) , (C,)
    N, D = X.shape

    # 计算前向传播
    scores = None
    #############################################################################
    # TODO: 执行前向传播, 计算从输入数据得到的每类的得分. #
    # 将结果存在scores变量, 形状为(N, C)。  #                                                      
    #############################################################################
    pass
    fc1= X.dot(W1) + b1 # (N,H)
    fc1_ReLU = np.maximum(fc1, 0) # ReLU函数
    fc2 = fc1_ReLU.dot(W2) + b2 # (N,C)
    scores = fc2 #(N,C)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # 如果没有给对应的正确标签，就跳出；结束
    if y is None:
      return scores

    # 计算loss
    loss = None
    #############################################################################
    # TODO: 完成前向传播，并且计算loss. 应该包含W1和W2的data loss和L2正则损失   #
    # 将结果存储在loss变量中，loss变量应该是一个标量                            #
    # 使用softmax损失函数                                                       #
    #############################################################################
    pass
    # softmax损失计算在第二层全连接层之后
    scores -=np.max(scores,axis=1,keepdims=True) # 每行数据减去每行的最大值，防止数值爆炸
    exp_scores = np.exp(scores)
    correct_exp_scores = exp_scores[np.arange(N),y]
    loss = -np.sum(np.log(correct_exp_scores/np.sum(exp_scores,axis=1).reshape(1,-1)))
    loss /= N #每个样本的平均损失
    loss += 0.5*reg*np.sum(W1*W1)+0.5*reg*np.sum(b1*b1) # 乘0.5的原因？
    loss += 0.5*reg*np.sum(W2*W2)+0.5*reg*np.sum(b2*b2)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # 反向传播: 计算梯度
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    pass
    dscore = exp_scores / exp_scores.sum(axis = 1, keepdims = True)
    dscore[range(N), y] -= 1
    dscore /= N
    dfc2 = dscore # dscore相当于loss对fc2（scores）的导数，dfc2,(N,C)
    dW2 =np.dot(fc1_ReLU.T,dfc2) + reg*W2
    dfc1_ReLU = (dfc2).dot(W2.T)
    db2 = np.sum(dfc2,axis = 0) #
    dfc1 = dfc1_ReLU 
    dfc1[fc1<=0] = 0 # relu对fc1的导数
    dW1 = X.T.dot(dfc1)+ reg*W1
    db1 = np.sum(dfc1,axis = 0) # 
   
    db2 += reg*b2
    db1 += reg*b1
     
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      index_batch = np.random.choice(X.shape[0],batch_size,replace=True) # 获取采样数据的索引
      X_batch = X[index_batch,:] #batch_size × D
      y_batch = y[index_batch]   #batch_size × 1
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      pass
      self.params['W1'] += -learning_rate * grads['W1']
      self.params['W2'] += -learning_rate * grads['W2']
      self.params['b1'] += -learning_rate * grads['b1']
      self.params['b2'] += -learning_rate * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    scores = np.maximum(X.dot(self.params['W1'])+self.params['b1'],0).dot(self.params['W2'])+ self.params['b2'] 
    scores_order_index = np.argsort(-scores,axis = 1) # 在每行中降序排序，返回降序类别索引
    y_pred = scores_order_index[:,0] #获取每行数据对应的分数最大的类别索引
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


