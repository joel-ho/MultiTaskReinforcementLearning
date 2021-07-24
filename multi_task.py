from copy import deepcopy
import pickle
import numpy as np
import matplotlib.pyplot as plt

from function_approximator import Actor, Critic
from experience_bank import ExperienceBank
from wrapped_env import CartPoleCustom, AcrobotCustom

np.random.seed(0)

# Settings
gamma = 0.99
n_td_steps = 2
replay_ratio = 4
n_episodes_task_switch = 500

entropy_coeff = 0.01
policy_clone_coeff = 0.005
value_clone_coeff = 0.001

n_episodes = 4000
r_avg_window = 100

# Environments
envs = [CartPoleCustom(), AcrobotCustom()]
n_states = envs[0].get_n_obs()
n_actions = envs[0].get_n_actions()
n_env_steps = [200, 200]

# Actor
func_pi = Actor(
    (n_states, 256, 256, n_actions), 
    ('relu', 'relu', 'softmax_cross_entropy'), 
    'cross_entropy_entropy_reg', 
    'adam', 
    64)

# Critic
func_V = Critic(
    (n_states, 256, 256, 1), 
    ('relu', 'relu', 'identity'), 
    'mean_squared_error_clone', 
    'adam', 
    64,
    use_lag=True)

# Experience
exp = ExperienceBank(int(1e7), n_td_steps, n_states, n_actions)

# Poisson distr
def poisson(lam, minimum=1, maximum=10):
  p = lam**np.arange(minimum, maximum+1)*np.exp(-lam)
  fac = np.math.factorial(minimum-1)
  for i, k in enumerate(range(minimum, maximum+1)):
    fac *= k
    p[i] /= fac
  delta = 1-np.sum(p)
  p += delta/(maximum+1-minimum)
  return minimum, maximum, p
  
def get_n_replay(lam):
  minimum, maximum, p = poisson(lam)
  return np.random.choice(np.arange(minimum, maximum+1), 1, p=p)[0]
  
# Training loop
r_all = [np.zeros((n_episodes, )), np.zeros((n_episodes, ))]
r_all_avg = [np.nan*np.ones((n_episodes, )), np.nan*np.ones((n_episodes, ))]
for i_episode in range(n_episodes):
   
  i_train = np.mod(int(i_episode/n_episodes_task_switch), 2)
  i_test = i_train - 1
  
  env_train = envs[i_train]
  env_test = envs[i_test]
  
  #############
  ### Train ###
  #############
  
  # Initialize state
  s = env_train.reset()

  # Sample trajectory
  for i_time in range(n_env_steps[i_train]):
    
    # Perform action
    pi = func_pi.predict(s[np.newaxis]) # 1 x n_actions
    V = func_V.predict(s[np.newaxis])[0, 0] # Scalar
    a = np.random.choice(n_actions, 1, p=pi[0, :])[0]
    s_prime, r, done, info = env_train.step(a)
    
    # Store in experience bank
    exp.write_sequentially(s, a, r, pi[0, :], V, s_prime, done)
    
    # Accumulate gradients
    if np.mod(i_time+1, n_td_steps) == 0:
      n_replay = get_n_replay(replay_ratio) # Int
      i_on_policy = [exp.get_latest_idx(), ] # List
      i_off_policy = list(exp.get_sample_idx(n_replay)) # List
      
      # Sample experience
      for i_exp in i_on_policy+i_off_policy:
      
        exp_s, exp_a, exp_r, exp_mu, exp_V, exp_s_prime, exp_terminal, exp_k = exp.get_traj(i_exp)
        
        Vcurr_prime = func_V.predict_from_lagged(exp_s_prime)[0, 0] # Scalar
        if exp_terminal:
          Vtr = 0
        else:
          pi = func_pi.predict(exp_s_prime) # 1 x n_actions
          Vtr = deepcopy(Vcurr_prime)
          
        # n-step off-policy TD
        for i in reversed(range(exp_k)):
          
          pi = func_pi.predict(exp_s[i:i+1, :]) # 1 x n_actions
          rho = pi/exp_mu[i:i+1, :] # 1 x n_actions
          rho_bar = np.min((1, rho[0, exp_a[i, 0]])) # scalar
          Vcurr = func_V.predict_from_lagged(exp_s[i:i+1, :])[0, 0] # scalar
          
          g1 = np.min((10, rho[0, exp_a[i, 0]]))*(exp_r[i, 0] + gamma*Vtr - Vcurr) # Scalar, Vtr is at i+1
          actor_train_y = np.zeros(pi.shape)
          actor_train_y[0, exp_a[i, 0]] += g1
          if i_exp > 0:
            actor_train_y += policy_clone_coeff*exp_mu[i:i+1, :]
          
          func_pi.accumulate_training_data(exp_s[i, :], actor_train_y, entropy_coeff)
          
          Vtr = rho_bar*(exp_r[i, 0] + gamma*Vtr) + (1 - rho_bar)*Vcurr # Scalar, Vtr at i
          
          if i_exp > 0:
            func_V.accumulate_training_data(exp_s[i, :], Vtr*np.ones((1, 1)), exp_V[i, 0], value_clone_coeff)
          else:
            func_V.accumulate_training_data(exp_s[i, :], Vtr*np.ones((1, 1)), 0, value_clone_coeff)
          
          Vcurr_prime = exp_r[i, 0] + gamma*Vcurr
    
    # Update total rewards and transition to next step
    r_all[i_train][i_episode] += r
    if done:
      break
    s = s_prime
    
    # End of training episode
    
  ############
  ### Test ###
  ############
  
  s = env_test.reset()
  for i_time in range(n_env_steps[i_test]):
  
    # Perform action
    pi = func_pi.predict(s[np.newaxis]) # 1 x n_actions
    a = np.random.choice(n_actions, 1, p=pi[0, :])[0]
    s_prime, r, done, info = env_test.step(a)
    
    r_all[i_test][i_episode] += r
    if done:
      break
    s = s_prime
    
    # End of testing episode
  
  #################
  ### Averaging ###
  #################
  
  if i_episode >= r_avg_window:
    for i_train_test in range(2):
      r_all_avg[i_train_test][i_episode] = np.mean(r_all[i_train_test][i_episode-r_avg_window:i_episode])
    
  if np.mod(i_episode+1, n_episodes_task_switch)==0:
    print('Episode {}, env_0_r_avg {}, env_1_r_avg {}, training {}'.format(i_episode+1, r_all_avg[0][i_episode], r_all_avg[1][i_episode], i_train))
    with open(('multi_task_data\\agent_%07d.p'%(i_episode+1)), 'wb') as f:
      pickle.dump([func_pi, func_V], f)
    with open(('residuals_%07d.p'%(i_episode+1)), 'wb') as f:
      pickle.dump([r_all, r_all_avg], f)
      
  if np.mod(i_episode+1, 2*n_episodes_task_switch)==0:
    with open('multi_task_data\\exp_bank.p', 'wb') as f:
      pickle.dump(exp, f)
  
envs[0].close()
envs[1].close()

plt.plot(np.arange(n_episodes), r_all_avg[0], 'b')
plt.plot(np.arange(n_episodes), r_all_avg[1], 'r')
plt.grid()

plt.show()