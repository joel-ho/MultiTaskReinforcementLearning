import numpy as np

from neural_net import NeuralNet

class FunctionApproximator(object):
  
  def __init__(self, 
      layers, activation, objective, optimizer, batch_size, 
      use_lag=False, lag_update_rate=50):
    self._nn = NeuralNet(layers, activation, objective, optimizer, batch_size)
    self._i_train = -1
    self._train_x = np.zeros((self._nn._batch_size, self._nn._layers[0]))
    self._train_y = np.zeros((self._nn._batch_size, self._nn._layers[-1]))
    self._use_lag = use_lag
    if use_lag:
      self._nn_lag = NeuralNet(layers, activation, objective, optimizer, batch_size)
      self._lag_update_rate = lag_update_rate
      self._i_lag_counter = 0
      self._lag_updated = False
    
  def _update_counter(self):
    self._i_train = int(np.mod(self._i_train+1, self._nn._batch_size))
    
  def _update_nn_weights(self, **kwargs):
    self._nn.compute_gradients(self._train_x, self._train_y, **kwargs)    
    self._nn.update_weights()
    if self._use_lag:
      self._i_lag_counter = int(np.mod(self._i_lag_counter+1, self._lag_update_rate))
      if self._i_lag_counter == 0:
        self.update_lagged_nn()
        self._lag_updated = True
    
  def get_nn(self):
    return self._nn
    
  def accumulate_training_data(self, x, y, auto_update=True):
    self._update_counter()
    self._train_x[self._i_train, :] = x
    self._train_y[self._i_train, :] = y    
    if (self._i_train==self._nn._batch_size-1) and auto_update:
      self._update_nn_weights()
      
  # def force_update_weight(self, **kwargs):
    # self._train_x[self._i_train+1:, :] = 0
    # self._train_y[self._i_train+1:, :] = 0
    # self.__update_nn_weights(**kwargs)
    # self._i_train = -1
    
  def update_lagged_nn(self):
    self._nn_lag.copy_from(self._nn)
      
  def predict(self, x):
    return self._nn.predict(x)
    
  def predict_from_lagged(self, x):
    if self._lag_updated:
      return self._nn_lag.predict(x)
    else:
      return np.zeros((x.shape[0], self._nn._layers[-1]))
      

class Actor(FunctionApproximator):
  
  def __init__(self, 
      layers, activation, objective, optimizer, batch_size, 
      use_lag=False, lag_update_rate=100):
      
    super().__init__(
        layers, activation, objective, optimizer, batch_size, 
        use_lag, lag_update_rate)
        
  def _update_nn_weights(self, **kwargs):
    self._nn.compute_gradients(self._train_x, self._train_y, **kwargs)
    # self._nn._optimizer_update_weights_adam(p=True)
    self._nn.update_weights()
    if self._use_lag:
      self._i_lag_counter = int(np.mod(self._i_lag_counter+1, self._lag_update_rate))
      if self._i_lag_counter == 0:
        self.update_lagged_nn()
        self._lag_updated = True
        
  def accumulate_training_data(self, x, y, reg_coeff, auto_update=True):
    self._update_counter()
    self._train_x[self._i_train, :] = x
    self._train_y[self._i_train, :] = y    
    if (self._i_train==self._nn._batch_size-1) and auto_update:
      self._update_nn_weights(ent_reg_coeff=reg_coeff)


class Critic(FunctionApproximator):
  
  def __init__(self, 
      layers, activation, objective, optimizer, batch_size, 
      use_lag=False, lag_update_rate=100):
      
    super().__init__(
        layers, activation, objective, optimizer, batch_size, 
        use_lag, lag_update_rate)
        
    self._train_y_clone = np.zeros(self._train_y.shape)
    
  def accumulate_training_data(self, x, y, Y_clone, clone_coeff, auto_update=True):
    self._update_counter()
    self._train_x[self._i_train, :] = x
    self._train_y[self._i_train, :] = y    
    self._train_y_clone[self._i_train, :] = Y_clone
    if (self._i_train==self._nn._batch_size-1) and auto_update:
      self._update_nn_weights(Y_clone=self._train_y_clone, clone_coeff=clone_coeff)