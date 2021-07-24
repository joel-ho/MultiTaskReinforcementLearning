from copy import deepcopy
import numpy as np

class ExperienceBank(object):

  def __init__(self, n_exp_total, n_td_steps, n_states, n_actions):
    
    self._n_exp_total = n_exp_total
    self._n_td_steps = n_td_steps
    self._n_states = n_states
    self._n_actions = n_actions
    
    self._i_exp_last_full = -1
    self._i_exp = -1
    self._i_traj = -1
    self._exp_bank_full = False
    
    self._init_memory()
    
  def _init_memory(self):
    self._states = np.zeros((self._n_exp_total, self._n_td_steps, self._n_states))
    self._actions = np.zeros((self._n_exp_total, self._n_td_steps, 1), dtype=int)
    self._policies = np.zeros((self._n_exp_total, self._n_td_steps, self._n_actions))
    self._rewards = np.zeros((self._n_exp_total, self._n_td_steps, 1))
    self._values = np.zeros((self._n_exp_total, self._n_td_steps, 1))
    self._next_states = np.zeros((self._n_exp_total, 1, self._n_states)) # Store next state for the last state in traj
    self._terminal = np.zeros((self._n_exp_total, ), dtype=bool)
    self._traj_count = np.zeros((self._n_exp_total, ), dtype=int)
    
  def _update_traj_counter(self):
    self._i_traj = int(np.mod(self._i_traj+1, self._n_td_steps))
    if self._i_traj == 0: # If current experience is full
      self._update_exp_counter()
    self._traj_count[self._i_exp] = self._i_traj+1
    
  def _update_exp_counter(self):
    self._i_exp_last_full = deepcopy(self._i_exp)
    if not(self._exp_bank_full):
      self._i_exp = int(np.mod(self._i_exp+1, self._n_exp_total))
    else:
      self._i_exp = np.random.choice(self._n_exp_total, 1)[0] # Randomly overwrite past experience
    
  def _check_bank_full(self):
    if not(self._exp_bank_full):
      if self._i_exp == self._n_exp_total-1:
        if self._i_traj == self._n_td_steps-1:
          self._exp_bank_full = True
    
  def get_n_exp_curr(self):
    if self._exp_bank_full:
      return self._n_exp_total
    else:
      return self._i_exp + 1
    
  def write_sequentially(self, state, action, reward, policy, value, next_state, terminal):
    
    # Store step
    self._update_traj_counter()
    for i, j in zip(
        (state, action, reward, policy, value), 
        (self._states, self._actions, self._rewards, self._policies, self._values)):
      j[self._i_exp, self._i_traj, :] = i
      
    self._next_states[self._i_exp, 0, :] = next_state
    self._terminal[self._i_exp] = terminal
    
    # Prepare to store in new experience for next step
    if terminal: 
      self._i_traj = -1
      
    self._check_bank_full()
      
  def get_traj(self, i_exp):
    return (
        self._states[i_exp, :self._traj_count[i_exp], :], 
        self._actions[i_exp, :self._traj_count[i_exp], :], 
        self._rewards[i_exp, :self._traj_count[i_exp], :], 
        self._policies[i_exp, :self._traj_count[i_exp], :],
        self._values[i_exp, :self._traj_count[i_exp], :], 
        self._next_states[i_exp, :, :],
        self._terminal[i_exp], 
        self._traj_count[i_exp]
        )
      
  def get_sample_idx(self, n):
    n_sample = min(n, self.get_n_exp_curr())
    return np.random.choice(self.get_n_exp_curr(), n_sample, False)
      
  def get_latest_idx(self):
    # Return the last full traj
    if (
        (self._i_traj==self._n_td_steps-1) or 
        (not(self._exp_bank_full) and self._i_exp==0)
        ):
      return self._i_exp
    else:
      return self._i_exp_last_full
      
  def copy_smaller_exp(self, exp):
    
    for i, j in zip(
        (exp._states, exp._actions, exp._rewards, exp._policies, exp._values, exp._next_states), 
        (self._states, self._actions, self._rewards, self._policies, self._values, self._next_states)):
      j[:exp._n_exp_total, :, :] = i
      j[exp._n_exp_total:, :, :] = 0
    
    self._terminal[:exp._n_exp_total] = exp._terminal
    self._terminal[exp._n_exp_total:] = 0
    self._traj_count[:exp._n_exp_total] = exp._traj_count
    self._traj_count[exp._n_exp_total:] = 0
    
    if exp._exp_bank_full:
      self._i_exp = exp._n_exp_total
    else:
      self._i_exp = exp.get_latest_idx()+1
      
    self._i_traj = -1
    self._exp_bank_full = False
      
if __name__ == '__main__':

  exp = ExperienceBank(5, 3, 2, 2)
  for i in range(1, 16):
    if i==8:
      terminal = True
    else:
      terminal = False
    exp.write_sequentially([i, i], np.random.choice(2, 1, p=[0.25, 0.75])[0], 0, [0.25, 0.75], 0.5, [i+1, i+1], terminal)
  
  print(exp._states)