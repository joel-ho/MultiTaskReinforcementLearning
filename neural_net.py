import numpy as np

class NeuralNet(object):

  def __init__(self, layers, activation, objective, optimizer, batch_size):
    
    self._n_layers = len(layers) # p layers
    self._layers = layers
    self._activation = activation
    self._objective = objective
    self._optimizer = optimizer
    self._batch_size = batch_size
    
    self._setup_checks()
    self._initialize_neurons()
    self._initialize_weights()
    self._initialize_activation_functions()
    self._initialize_objective_function()
    self._initialize_optimizer()
  
  #######################
  ### Initializations ###
  #######################
  def _setup_checks(self):
    
    # Checks to ensure softmax is used correctly
    for i, act in enumerate(self._activation):
      if not(i == self._n_layers-2) and (act.lower() == 'softmax_cross_entropy'):
        raise Exception('softmax_cross_entropy can only be used on last layer.')
  
    if (
        ((self._objective.lower() == 'cross_entropy') or (self._objective.lower() == 'cross_entropy_entropy_reg')) 
        and 
        not(self._activation[-1] == 'softmax_cross_entropy')
        ):
      raise Exception('softmax_cross_entropy activation has to be used with cross_entropy')
      
    if (
        not((self._objective.lower() == 'cross_entropy') or (self._objective.lower() == 'cross_entropy_entropy_reg')) 
        and 
        (self._activation[-1] == 'softmax_cross_entropy')
        ):
      raise Exception('softmax_cross_entropy activation has to be used with cross_entropy')
  
  def _initialize_neurons(self):
    self._a = []          # List of length p
    self._b = [0, ]       # List of length p
    self._b_grad = [0, ]  # List of length p
    self._Z = [0, ]       # List of length p
    self._delta = [0, ]   # List of length p
    self._a_prime = [0, ] # List of length p
    for i in range(self._n_layers):      
      self._a.append(np.zeros(
          (1, self._layers[i], self._batch_size)
          ))
      if i > 0:
        self._Z.append(np.zeros(
            (self._layers[i], self._batch_size)
            ))
        self._delta.append(np.zeros(
            (1, self._layers[i], self._batch_size)
            ))
        self._b.append(np.zeros(
            (self._layers[i], 1)
            ))
        self._b_grad.append(np.zeros(
            (self._layers[i], 1)
            ))
        self._a_prime.append(np.zeros(
            (self._layers[i], self._batch_size)
            ))
          
  def _initialize_weights(self):
    self._W = []            # List of length (p-1)
    self._W_grad_bat = []   # List of length (p-1)
    self._W_grad = []       # List of length (p-1)
    for i in range(self._n_layers-1):
      self._W.append(np.random.random(
          (self._layers[i+1], self._layers[i])
          )-0.5)
      self._W_grad_bat.append(np.zeros(
          (self._batch_size, self._layers[i+1], self._layers[i])
          ))
      self._W_grad.append(np.zeros(
          (self._layers[i+1], self._layers[i])
          ))
  
  ############################
  ### Activation functions ###
  ############################
  def _initialize_activation_functions(self):
    compute_act_fun_map = {
        'relu': self._compute_activation_relu,
        'identity': self._compute_activation_identity,
        'softmax_cross_entropy': self._compute_activation_softmax_cross_entropy 
        }
    compute_act_fun_deriv_map = {
        'relu': self._compute_activation_derivative_relu,
        'identity': self._compute_activation_derivative_identity,
        'softmax_cross_entropy': self._compute_activation_derivative_softmax_cross_entropy
        }
    self._compute_act_fun_list = [0, ] # Keep indexing consistent (size p)
    self._compute_act_fun_deriv_list = [0, ] # Keep indexing consistent (size p)
    for i, act in enumerate(self._activation):   
      self._compute_act_fun_list.append(
          compute_act_fun_map[act.lower()]
          )
      self._compute_act_fun_deriv_list.append(
          compute_act_fun_deriv_map[act.lower()]
          )
  
  def _compute_activation_relu(self, Z, train=False, i_lay=None):
    if train: # Write into preallocated array
      self._a[i_lay][0, :, :] = self._Z[i_lay]
      self._a[i_lay][0, self._Z[i_lay]<0] *= 0.01
    else:
      a = Z
      a[Z<0] *= 0.01
      return a
    
  def _compute_activation_derivative_relu(self, i_lay):
    self._a_prime[i_lay][:, :] = 1
    self._a_prime[i_lay][self._Z[i_lay]<0] = 0.01
    
  def _compute_activation_softmax_cross_entropy(self, Z, train=False, i_lay=None):
    if train: # Write into preallocated array
      e = np.exp(self._Z[i_lay]-np.max(self._Z[i_lay], axis=0))
      self._a[i_lay][0, :, :] = e/np.sum(e, axis=0)
    else:
      e = np.exp(Z-np.max(Z, axis=0))
      return e/np.sum(e)
  
  def _compute_activation_derivative_softmax_cross_entropy(self, i_lay):
    # Only works with softmax at final layer and cross entropy
    self._a_prime[i_lay][:, :] = 1
  
  def _compute_activation_identity(self, Z, train=False, i_lay=None):
    if train: # Write into preallocated array
      self._a[i_lay][0, :, :] = self._Z[i_lay]
    else:
      return Z
    
  def _compute_activation_derivative_identity(self, i_lay):
    self._a_prime[i_lay][:, :] = 1
  
  ###########################
  ### Objective functions ###
  ###########################
  def _initialize_objective_function(self):
    compute_objective_fun_map = {
        'mean_squared_error': self._compute_objective_function_mse,
        'mean_squared_error_clone': self._compute_objective_function_mse_clone,
        'cross_entropy': self._compute_objective_function_cross_entropy,
        'cross_entropy_entropy_reg': self._compute_objective_function_cross_entropy_entropy_reg
        }
    compute_objective_fun_deriv_map = {
        'mean_squared_error': self._compute_objective_function_derivative_mse,
        'mean_squared_error_clone': self._compute_objective_function_derivative_mse_clone,
        'cross_entropy': self._compute_objective_function_derivative_cross_entropy,
        'cross_entropy_entropy_reg': self._compute_objective_function_derivative_cross_entropy_entropy_reg
        }
    self._compute_obj_fun = compute_objective_fun_map[self._objective.lower()]
    self._compute_obj_fun_deriv = compute_objective_fun_deriv_map[self._objective.lower()]
  
  def _compute_objective_function_mse(self, Y, Y_correct, **kwargs):
    sqerr = (Y-Y_correct)**2
    return 1/(2*Y.shape[0])*sqerr.sum()
    
  def _compute_objective_function_derivative_mse(self, Y, Y_correct, **kwargs):
    return 1/(Y.shape[0])*(Y-Y_correct)
  
  def _compute_objective_function_mse_clone(self, Y, Y_correct, **kwargs):
    err = (Y-Y_correct)**2 + kwargs['clone_coeff']*(Y-kwargs['Y_clone'])**2
    return 1/(2*Y.shape[0])*err.sum()
    
  def _compute_objective_function_derivative_mse_clone(self, Y, Y_correct, **kwargs):
    return 1/(Y.shape[0])*(Y-Y_correct + kwargs['clone_coeff']*(Y-kwargs['Y_clone']))
  
  def _compute_objective_function_cross_entropy(self, Y, Y_correct, **kwargs):
    # Y is probabilities output by model
    dot_prod = -Y_correct*np.log(Y)
    return 1/(Y.shape[0])*dot_prod.sum()
  
  def _compute_objective_function_derivative_cross_entropy(self, Y, Y_correct, **kwargs):
    # Only works with softmax at final layer and cross entropy 
    return (1/Y.shape[0])*(
        self._a[self._n_layers-1][0, :, :].T*np.sum(Y_correct, axis=1)[np.newaxis].T - 
        Y_correct)
  
  def _compute_objective_function_cross_entropy_entropy_reg(self, Y, Y_correct, **kwargs):
    # Cross entropy loss with entropy regularization
    # Need to pass in ent_reg_coeff as kwarg
    dot_prod = (-Y_correct + kwargs['ent_reg_coeff']*Y)*np.log(Y)
    return 1/(Y.shape[0])*dot_prod.sum()
  
  def _compute_objective_function_derivative_cross_entropy_entropy_reg(self, Y, Y_correct, **kwargs):
    # Cross entropy loss with entropy regularization
    # Need to pass in ent_reg_coeff as kwarg
    a = self._a[self._n_layers-1][0, :, :].T
    deriv = a*np.sum(Y_correct, axis=1)[np.newaxis].T - Y_correct
    for i in range(Y.shape[1]):
      for j in range(Y.shape[1]):
        if j == i:
          deriv[:, i] += kwargs['ent_reg_coeff']*(np.log(a[:, i])-1)*a[:, i]*(1-a[:, i])
        else:
          deriv[:, i] += kwargs['ent_reg_coeff']*(np.log(a[:, j])-1)*(-a[:, i]*a[:, j])
    return deriv/Y.shape[0]
  
  #################
  ### Optimizer ###
  #################
  def _initialize_optimizer(self):
  
    # Instantiate correct optimizer
    optimizer_initialization_map = {
        'adam': self._initialize_optimizer_adam,
        'sgd': self._initialize_optimizer_sgd}
    optimizer_update_weights_map = {
        'adam': self._optimizer_update_weights_adam,
        'sgd': self._optimizer_update_weights_sgd}
    optimizer_compare_map = {
        'adam': self._compare_optimizer_adam,
        'sgd': self._compare_optimizer_sgd}
    
    optimizer_initialization_map[self._optimizer.lower()]() # Initialize required arrays
    self._optimizer_update_W_fun = optimizer_update_weights_map[self._optimizer.lower()]
    self._optimizer_comparison_fun = optimizer_compare_map[self._optimizer.lower()]
  
  def _initialize_optimizer_adam(self):
    
    self._adam_beta1 = 0.9
    self._adam_beta2 = 0.999
    self._adam_eta = 0.001
    self._adam_eps = 1e-8
    
    self._adam_m_b = [0, ]  # List of length (p) for adam optimizer
    self._adam_v_b = [0, ]  # List of length (p) for adam optimizer
    self._adam_m_W = []     # List of length (p-1) for adam optimizer
    self._adam_v_W = []     # List of length (p-1) for adam optimizer
    
    for i in range(1, self._n_layers):
      self._adam_m_W.append(np.zeros(
          (self._layers[i], self._layers[i-1])
          ))
      self._adam_v_W.append(np.zeros(
          (self._layers[i], self._layers[i-1])
          ))
      self._adam_m_b.append(np.zeros(
          (self._layers[i], 1)
          ))
      self._adam_v_b.append(np.zeros(
          (self._layers[i], 1)
          ))
  
  def _optimizer_update_weights_adam(self, p=False):
    
    if p:
      delta_W = []
      delta_b = []
    for i_lay in range(self._n_layers-1):
      
      self._adam_m_W[i_lay] *= self._adam_beta1
      self._adam_m_W[i_lay] += (1-self._adam_beta1)*self._W_grad[i_lay]
      self._adam_v_W[i_lay] *= self._adam_beta2
      self._adam_v_W[i_lay] += (1-self._adam_beta2)*self._W_grad[i_lay]**2
      
      self._adam_m_b[i_lay] *= self._adam_beta1
      self._adam_m_b[i_lay] += (1-self._adam_beta1)*self._b_grad[i_lay]
      self._adam_v_b[i_lay] *= self._adam_beta2
      self._adam_v_b[i_lay] += (1-self._adam_beta2)*self._b_grad[i_lay]**2
    
      self._W[i_lay] -= self._adam_eta*(
          self._adam_m_W[i_lay]/(1-self._adam_beta1)/ # m_hat
          ( np.sqrt(self._adam_v_W[i_lay]/(1-self._adam_beta2)) + self._adam_eps ) # sqrt(v_hat) + eps
          )
      self._b[i_lay] -= self._adam_eta*(
          self._adam_m_b[i_lay]/(1-self._adam_beta1)/ # m_hat
          ( np.sqrt(self._adam_v_b[i_lay]/(1-self._adam_beta2)) + self._adam_eps ) # sqrt(v_hat) + eps
          )
      if p:    
        delta_W.append(-self._adam_eta*(
            self._adam_m_W[i_lay]/(1-self._adam_beta1)/ # m_hat
            ( np.sqrt(self._adam_v_W[i_lay]/(1-self._adam_beta2)) + self._adam_eps ) # sqrt(v_hat) + eps
            )
        )
        delta_b.append(-self._adam_eta*(
            self._adam_m_b[i_lay]/(1-self._adam_beta1)/ # m_hat
            ( np.sqrt(self._adam_v_b[i_lay]/(1-self._adam_beta2)) + self._adam_eps ) # sqrt(v_hat) + eps
            )
        )
    if p:
      for i, g in enumerate(delta_W+delta_b):
        if i==0:
          g_flat = g.flatten()
        else:
          g_flat = np.hstack((g_flat, g.flatten()))
          
      with open('33999_alternate_gradients', 'ab') as f:
        np.savetxt(f, g_flat[np.newaxis])
      
  def _compare_optimizer_adam(self, nn):
    
    if not(self._adam_beta1 == nn._adam_beta1):
      return False
    if not(self._adam_beta2 == nn._adam_beta2):
      return False
    if not(self._adam_eta == nn._adam_eta):
      return False
    if not(self._adam_eps == nn._adam_eps):
      return False
    
    return True
  
  def _initialize_optimizer_sgd(self):
    self._sgd_update_rate = 0.01
    
  def _optimizer_update_weights_sgd(self):
    for i_lay in range(self._n_layers-1):
      self._W[i_lay] -= self._sgd_update_rate*self._W_grad[i_lay]
      self._b[i_lay] -= self._sgd_update_rate*self._b_grad[i_lay]
  
  def _compare_optimizer_sgd(self, nn):
    if not(self._sgd_update_rate == nn._sgd_update_rate):
      return False
    
    return True
  
  ####################
  ### NN functions ###
  ####################
  def create_mini_batch(self, X_train, Y_train, batch_size):
    
    m = X_train.shape[0]
    n_batches = int(np.ceil(m/batch_size))
    scramble_mask = np.random.choice(m, m, False)
    
    X_train_scram = X_train[scramble_mask, :]
    Y_train_scram = Y_train[scramble_mask, :]
    X_train_batch = []
    Y_train_batch = []
    for i in range(n_batches):
      if i < n_batches-1:
        X_train_batch.append(
            X_train_scram[i*batch_size:(i+1)*batch_size]
            )
        Y_train_batch.append(
            Y_train_scram[i*batch_size:(i+1)*batch_size]
            )
      else:
        n_remain = m-batch_size*(n_batches-1)
        X_train_batch.append(X_train_scram[-n_remain:])
        Y_train_batch.append(Y_train_scram[-n_remain:])
        
    return X_train_batch, Y_train_batch
  
  def compute_gradients(self, X, Y_correct, **kwargs):
    # X is size (batch_size x n1) where m is number of points to calculate 
    # for and n1 is size of input 
    # kwargs for objective function
    
    self._a[0][0, :, :] = X.T
    
    # Forward prop
    for i_lay in range(1, self._n_layers): # From second layer to the end
      np.matmul(self._W[i_lay-1], self._a[i_lay-1][0, :, :], out=self._Z[i_lay]) 
      self._Z[i_lay] += self._b[i_lay]
      self._compute_act_fun_list[i_lay](None, True, i_lay)
    Y = self._a[-1][0, :, :].T
    obj_value = self._compute_obj_fun(Y, Y_correct, **kwargs)
    
    # Back prop
    self._compute_act_fun_deriv_list[-1](self._n_layers-1)
    self._delta[-1][0, :, :] = ( # Last layer
        self._compute_obj_fun_deriv(Y, Y_correct, **kwargs).T*
        self._a_prime[i_lay]
        )
    for i_lay in reversed(range(1, self._n_layers-1)):
      self._compute_act_fun_deriv_list[i_lay](i_lay)
      np.matmul(self._W[i_lay].T, self._delta[i_lay+1][0, :, :], out=self._delta[i_lay][0, :, :])
      self._delta[i_lay][0, :, :] *= self._a_prime[i_lay]
    
    # Compute gradients
    for i_lay in range(1, self._n_layers): # From second layer to the end
      # W_grad_batch[i] size: (batch_size x n_i+1 x n_i)
      self._W_grad_bat[i_lay-1][:, :, :] = np.matmul( 
          np.moveaxis(self._delta[i_lay], [0, 1, 2], [2, 1, 0]), 
          np.moveaxis(self._a[i_lay-1], [0, 1, 2], [1, 2, 0])
          )
      self._W_grad[i_lay-1][:, :] = np.sum( # Sum across batch size
          self._W_grad_bat[i_lay-1]
          , 0)
      self._b_grad[i_lay][:, :] = np.sum(self._delta[i_lay], axis=2).T
      
    return obj_value
  
  def update_weights(self):
    self._optimizer_update_W_fun()
    
  def predict(self, X, **kwargs):
    # X is size (m x n1) where m is number of points to calculate for and n1 is 
    # size of input
    Y = X.T
    for i_lay in range(1, self._n_layers):
      
      if 'write' in kwargs:
        if kwargs['write']:
          with open('{}_Z_{}'.format(kwargs['write_file_name'], i_lay), kwargs['write_modifier']) as f:
            np.savetxt(f, (np.matmul(self._W[i_lay-1], Y) + self._b[i_lay]).T)
            
      Y = self._compute_act_fun_list[i_lay](
          np.matmul(self._W[i_lay-1], Y) + self._b[i_lay]
          )
      
      if 'write' in kwargs:
        if kwargs['write']:
          with open('{}_a_{}'.format(kwargs['write_file_name'], i_lay), kwargs['write_modifier']) as f:
            np.savetxt(f, Y.T)
      
    return Y.T
    
  def get_flattened_weights(self):
    for i, W in enumerate(self._W+self._b):
      if i==0:
        W_flat = W.flatten()
      else:
        W_flat = np.hstack((W_flat, W.flatten()))
    return W_flat
    
  def get_flattened_gradients(self):
    for i, g in enumerate(self._W_grad+self._b_grad[1:]):
      if i==0:
        g_flat = g.flatten()
      else:
        g_flat = np.hstack((g_flat, g.flatten()))
    return g_flat
  
  def write_weights(self, write_name, write_modifier):
    with open('{}_weights'.format(write_name), write_modifier) as f:
        np.savetxt(f, self.get_flattened_weights()[np.newaxis])
    
  def write_gradients(self, write_name, write_modifier):
    with open('{}_weights_grad'.format(write_name), write_modifier) as f:
        np.savetxt(f, self.get_flattened_gradients()[np.newaxis])
    
  def compare_setup(self, nn):
  
    # Compare number of layers
    if not(self._n_layers == nn._n_layers):
      return False
  
    # Compare layer size
    for lay_self, lay in zip(self._layers, nn._layers):
      if not(lay_self == lay):
        return False
    
    # Compare activation functions
    for act_self, act in zip(self._activation, nn._activation):
      if not(act_self.lower() == act.lower()):
        return False
        
    # Check objective function
    if not(self._objective.lower() == nn._objective.lower()):
      return False
      
    # Check batch size
    if not(self._batch_size == nn._batch_size):
      return False
      
    # Check optimizer
    if not(self._optimizer.lower() == nn._optimizer.lower()):
      return False
    else:
      if not(self._optimizer_comparison_fun(nn)): # _optimizer_comparison_fun returns False if not the same
        return False
        
    return True
    
  def copy_from(self, nn):
    
    if not(self.compare_setup(nn)):
      return False
    
    for v_name in dir(self):
      v = getattr(self, v_name)
      v_copy = getattr(nn, v_name)
      if type(v) == list:
        for e, e_copy in zip(v, v_copy):
          if type(e) == np.ndarray:
            np.copyto(e, e_copy)
    
    return True
    
  ###################
  ### Diagnostics ###
  ###################
  def check_gradients(self, X, Y_correct):
    
    perturb_eps = 1e-8
    
    Y = self.predict(X)
    obj_val_ref = self._compute_obj_fun(Y, Y_correct)
    self.compute_gradients(X, Y_correct)
    
    # Perturb weights with finite difference
    for i_lay in range(self._n_layers-1):
      print('\n--- Weights layer {} ---'.format(i_lay))
      for i_row in range(self._W[i_lay].shape[0]):
        for i_col in range(self._W[i_lay].shape[1]):
          
          back_prop_gradient = self._W_grad[i_lay][i_row, i_col]
        
          self._W[i_lay][i_row, i_col] += perturb_eps
          Y = self.predict(X)
          obj_val_perturb = self._compute_obj_fun(Y, Y_correct)
          finite_diff_gradient = (obj_val_perturb-obj_val_ref)/perturb_eps
          print('Row {}, col {} --- bp: {}, fd: {}, delta: {}'.format(
              i_row, i_col, back_prop_gradient, finite_diff_gradient, 
              (finite_diff_gradient-back_prop_gradient)))
          
          self._W[i_lay][i_row, i_col] -= perturb_eps
  
    # Perturb biases
    for i_lay in range(1, self._n_layers):
      print('\n--- Bias layer {} ---'.format(i_lay))
      for i_neuron in range(self._layers[i_lay]):
        back_prop_gradient = self._b_grad[i_lay][i_neuron, 0]
        self._b[i_lay][i_neuron, 0] += perturb_eps
        Y = self.predict(X)
        obj_val_perturb = self._compute_obj_fun(Y, Y_correct)
        finite_diff_gradient = (obj_val_perturb-obj_val_ref)/perturb_eps
        print('Neuron {} --- bp: {}, fd: {}, delta: {}'.format(
            i_neuron, back_prop_gradient, finite_diff_gradient, 
            (finite_diff_gradient-back_prop_gradient)))
        self._b[i_lay][i_neuron, 0] -= perturb_eps
        
  def get_np_array_addr(self):
    addr = dict()
    get_np_addr = lambda x: x.__array_interface__['data'][0]
    for v_name in dir(self):
      v = getattr(self, v_name)
      if type(v) == list:
        for i, e in enumerate(v):
          if type(e) == np.ndarray:
            k = '{}[{}]'.format(v_name, i)
            addr[k] = get_np_addr(e)
    return addr
    
    
############
### Test ###
############
if __name__ == '__main__':
  
  import matplotlib.pyplot as plt
  
  run_tests = True
  
  def check_numpy_addr():
    
    x = np.linspace(-1, 1, 32)[np.newaxis].T
    y1 = x**2-0.5
    y2 = 0.5*x -0.3*x**3 + 0.2
    y = np.hstack((y1, y2))
    
    nn = NeuralNet((1, 8, 8, 8, 2), ('relu', 'relu', 'relu', 'identity'), 'mean_squared_error', 'adam', 32)
    org_addr = nn.get_np_array_addr()
    
    obj_value = nn.compute_gradients(x, y)
    nn.update_weights()
    
    check_addr = nn.get_np_array_addr()  
    for k, v in org_addr.items():
      print('{}, {}, {}, delta: {}'.format(k, v, check_addr[k], (v-check_addr[k])))
  
  def check_gradients():
  
    x = np.linspace(-1, 1, 32)[np.newaxis].T
    y1 = x**2-0.5
    y2 = 0.5*x -0.3*x**3 + 0.2
    y = np.hstack((y1, y2))
    
    nn = NeuralNet((1, 3, 3, 2), ('relu', 'relu', 'identity'), 'mean_squared_error', 'adam', 32)
    
    nn.check_gradients(x, y)
  
  def check_gradients_softmax():
  
    x = np.linspace(-1, 1, 8)[np.newaxis].T
    y = np.zeros((8, 3))
    
    y[:2, 0] = np.random.random((2, ))
    y[:2, 2] = 1 - np.random.random((2, ))
    
    y[2:6, 1] = np.random.random((4, ))
    y[2:6, 0] = 0.5*(1-y[2:6, 1])
    y[2:6, 2] = 0.5*(1-y[2:6, 1])
    
    y[6:, 2] = 0
    
    nn = NeuralNet(
        (1, 3, 3, 3), 
        ('relu', 'relu', 'softmax_cross_entropy'), 
        'cross_entropy', 
        'adam', 
        8)
    
    nn.check_gradients(x, y)
    
  def check_setup_softmax():
    
    print('\n\n--- Checking softmax setup checks ---')
    try:
      nn = NeuralNet(
          (1, 3, 3, 3), 
          ('relu', 'softmax_cross_entropy', 'relu'), 
          'mean_squared_error', 
          'adam', 
          32)
      print('Test failed.')
    except:
      print('Exception raised successfully.')
    
    try:
      nn = NeuralNet(
          (1, 3, 3, 3), 
          ('relu', 'relu', 'identity'), 
          'cross_entropy', 
          'adam', 
          32)
      print('Test failed.')
    except:
      print('Exception raised successfully.')
  
    try:
      nn = NeuralNet(
          (1, 3, 3, 3), 
          ('relu', 'relu', 'softmax_cross_entropy'), 
          'mean_squared_error', 
          'adam', 
          32)
      print('Test failed.')
    except:
      print('Exception raised successfully.')
  
    print('-----------------------------------\n\n')
  
  def check_nn():
    
    x = np.linspace(-1, 1, 2000)[np.newaxis].T
    y1 = x**2-0.5
    y2 = 0.5*x -0.3*x**3 + 0.2
    y = np.hstack((y1, y2))
    
    nn = NeuralNet((1, 32, 32, 2), ('relu', 'relu', 'identity'), 'mean_squared_error', 'adam', 32)
    
    n_epoch = 100
    obj_value_vec = np.zeros((n_epoch, ))
    for i_epoch in range(n_epoch):
      x_train_batch, y_train_batch = nn.create_mini_batch(x, y, 32)
      for i_step in range(len(x_train_batch)-1): # Ignore last batch which may not be batch_size
        obj_value = nn.compute_gradients(x_train_batch[i_step], y_train_batch[i_step])
        if i_step == 0:
          print('Epoch: {}, Obj value: {}'.format((i_epoch+1), obj_value))
        nn.update_weights()
      obj_value_vec[i_epoch] = obj_value
    
    y_pred = nn.predict(x)
    
    plt.figure()
    plt.plot(np.arange(obj_value_vec.shape[0]), obj_value_vec)
    plt.grid()
    
    plt.figure()
    plt.plot(x, y1, 'b')
    plt.plot(x, y_pred[:, 0], 'r')
    plt.grid()
    
    plt.figure()
    plt.plot(x, y2, 'b')
    plt.plot(x, y_pred[:, 1], 'r')
    plt.grid()
    
    plt.show()
    
  def check_nn_softmax():
  
    x = np.linspace(-1, 1, 3000)[np.newaxis].T
    y = np.zeros((3000, 3))
    y[:1000, 0] = 1
    y[1000:2000, 1] = 1
    y[2000:, 2] = 1
    
    nn = NeuralNet(
        (1, 24, 24, 3), 
        ('relu', 'relu', 'softmax_cross_entropy'), 
        'cross_entropy', 
        'adam', 
        32)
    
    n_epoch = 50
    obj_value_vec = np.zeros((n_epoch, ))
    for i_epoch in range(n_epoch):
      x_train_batch, y_train_batch = nn.create_mini_batch(x, y, 32)
      for i_step in range(len(x_train_batch)-1): # Ignore last batch which may not be batch_size
        obj_value = nn.compute_gradients(x_train_batch[i_step], y_train_batch[i_step])
        if i_step == 0:
          print('Epoch: {}, Obj value: {}'.format((i_epoch+1), obj_value))
        nn.update_weights()
      obj_value_vec[i_epoch] = obj_value
    
    y_pred = nn.predict(x)
    
    plt.figure()
    plt.plot(x, np.argmax(y, axis=1), 'b')
    plt.plot(x, np.argmax(y_pred, axis=1), 'r')
    plt.grid()
    
    plt.show()
    
  def check_compare_setup():
    
    nn1 = NeuralNet((1, 8, 8, 8, 2), ('relu', 'relu', 'relu', 'identity'), 'mean_squared_error', 'adam', 32)
    nn2 = NeuralNet((1, 8, 8, 2), ('relu', 'relu', 'identity'), 'mean_squared_error', 'adam', 32)
    nn3 = NeuralNet((1, 8, 12, 8, 2), ('relu', 'relu', 'relu', 'identity'), 'mean_squared_error', 'adam', 32)
    nn4 = NeuralNet((1, 8, 8, 8, 2), ('relu', 'relu', 'identity', 'identity'), 'mean_squared_error', 'adam', 32)
    nn5 = NeuralNet((1, 8, 8, 8, 2), ('relu', 'relu', 'relu', 'identity'), 'mean_squared_error', 'adam', 16)
    nn6 = NeuralNet((1, 8, 8, 8, 2), ('relu', 'relu', 'relu', 'identity'), 'mean_squared_error', 'adam', 32)
    nn6._adam_beta1 = 0.8
    for i, nn in enumerate([nn1, nn2, nn3, nn4, nn5, nn6]):
      print('nn{}: {}'.format((i+1), nn1.compare_setup(nn)))
    
  def check_copy_nn():
    
    x = np.linspace(-1, 1, 4)[np.newaxis].T
    y1 = x**2-0.5
    y2 = 0.5*x -0.3*x**3 + 0.2
    y = np.hstack((y1, y2))
    
    nn1 = NeuralNet((1, 3, 3, 2), ('relu', 'relu', 'identity'), 'mean_squared_error', 'adam', 4)
    nn2 = NeuralNet((1, 3, 3, 2), ('relu', 'relu', 'identity'), 'mean_squared_error', 'adam', 4)
    for nn in [nn1, nn2]:
      obj_value = nn.compute_gradients(x, y)
      nn.update_weights()
    org_addr = nn2.get_np_array_addr()  
    
    print('\n--- Before copy ---')
    for i_lay in range(nn1._n_layers):
      print('_a[{}], sum of deltas: {}'.format(i_lay, (nn1._a[i_lay] - nn2._a[i_lay]).sum()))      
      if i_lay > 0:
        print('_Z[{}], sum of deltas: {}'.format(i_lay, (nn1._Z[i_lay] - nn2._Z[i_lay]).sum()))
        print('_b[{}], sum of deltas: {}'.format(i_lay, (nn1._b[i_lay] - nn2._b[i_lay]).sum()))
        print('_b_grad[{}], sum of deltas: {}'.format(i_lay, (nn1._b_grad[i_lay] - nn2._b_grad[i_lay]).sum()))
      if i_lay < nn1._n_layers-1:
        print('_W[{}], sum of deltas: {}'.format(i_lay, (nn1._W[i_lay] - nn2._W[i_lay]).sum()))
        print('_W_grad[{}], sum of deltas: {}'.format(i_lay, (nn1._W_grad[i_lay] - nn2._W_grad[i_lay]).sum()))
    
    nn2.copy_from(nn1)
    print('\n--- After copy ---')
    for i_lay in range(nn1._n_layers):
      print('_a[{}], sum of deltas: {}'.format(i_lay, (nn1._a[i_lay] - nn2._a[i_lay]).sum()))      
      if i_lay > 0:
        print('_Z[{}], sum of deltas: {}'.format(i_lay, (nn1._Z[i_lay] - nn2._Z[i_lay]).sum()))
        print('_b[{}], sum of deltas: {}'.format(i_lay, (nn1._b[i_lay] - nn2._b[i_lay]).sum()))
        print('_b_grad[{}], sum of deltas: {}'.format(i_lay, (nn1._b_grad[i_lay] - nn2._b_grad[i_lay]).sum()))
      if i_lay < nn1._n_layers-1:
        print('_W[{}], sum of deltas: {}'.format(i_lay, (nn1._W[i_lay] - nn2._W[i_lay]).sum()))
        print('_W_grad[{}], sum of deltas: {}'.format(i_lay, (nn1._W_grad[i_lay] - nn2._W_grad[i_lay]).sum()))
  
    print('\n--- Numpy addresses after copy ---')
    check_addr = nn2.get_np_array_addr()  
    for k, v in org_addr.items():
      print('{}, {}, {}, delta: {}'.format(k, v, check_addr[k], (v-check_addr[k])))
  
  def check_write_grad():
    x = np.linspace(-1, 1, 32)[np.newaxis].T
    y1 = x**2-0.5
    y2 = 0.5*x -0.3*x**3 + 0.2
    y = np.hstack((y1, y2))
    
    nn = NeuralNet((1, 3, 3, 2), ('relu', 'relu', 'identity'), 'mean_squared_error', 'adam', 32)
    obj_value = nn.compute_gradients(x, y)
    
    nn.write_gradients('test_gradient', 'wb')
    
    print('W_grad')
    for i in nn._W_grad:
      print('\n------------------------')
      print(i)
      print('\n------------------------')
    
    print('b_grad')  
    for i in nn._b_grad:
      print('\n------------------------')
      print(i)
      print('\n------------------------')
  
  def check_write_act():
    
    x = np.linspace(-1, 1, 4)[np.newaxis].T
    y1 = x**2-0.5
    y2 = 0.5*x -0.3*x**3 + 0.2
    y = np.hstack((y1, y2))
    
    nn = NeuralNet((1, 3, 3, 2), ('relu', 'relu', 'identity'), 'mean_squared_error', 'adam', 4)
    obj_value = nn.compute_gradients(x, y)
    nn.update_weights()
    
    y_pred = nn.predict(x, **{'write': True, 'write_file_name': 'test_act', 'write_modifier': 'wb'})
    print('y_pred')
    print(y_pred)
  
  # Call functions
  if run_tests:
    check_numpy_addr()
    check_gradients()
    check_gradients_softmax()
    check_setup_softmax()
    check_nn()
    check_nn_softmax()
    check_compare_setup()
    check_copy_nn()
    check_write_grad()
    check_write_act()
  