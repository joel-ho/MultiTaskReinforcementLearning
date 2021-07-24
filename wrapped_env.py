import numpy as np
import gym

class CartPoleCustom(object):
  def __init__(self):
    self._env = gym.make('CartPoleCustom-v0')
  
  def _s_org_to_transformed(self, s):
    return np.array([s[0]/4*0.5-0.25, s[1]/4*0.5-0.25, s[2]/0.4*0.5-0.25, s[3]/5*0.5-0.25])
  
  def get_n_obs(self):
    return self._env.observation_space.shape[0]
    
  def get_n_actions(self):
    return self._env.action_space.n
    
  def reset(self):
    s_org = self._env.reset()
    return self._s_org_to_transformed(s_org)
    
  def step(self, a):
    s_prime_org, r, done, info = self._env.step(a)
    s_prime_transformed = self._s_org_to_transformed(s_prime_org)
    return (s_prime_transformed, r, done, info)
    
  def close(self):
    self._env.close()
    

class AcrobotCustom(object):
  def __init__(self):
    self._env = gym.make('Acrobot-v1')
    
  def _s_org_to_transformed(self, s):
    return np.array([np.arccos(s[0])/3*0.5, np.arccos(s[2])/3*0.5, s[4]/10*0.5+0.25, s[5]/16*0.5+0.25])
    
  def get_n_obs(self):
    return 4
  
  def get_n_actions(self):
    return self._env.action_space.n
    
  def reset(self):
    s_org = self._env.reset()
    return self._s_org_to_transformed(s_org)
    
  def step(self, a):
    s_prime_org, r, done, info = self._env.step(a)
    s_prime_transformed = self._s_org_to_transformed(s_prime_org)
    return (s_prime_transformed, r, done, info)
    
  def close(self):
    self._env.close()