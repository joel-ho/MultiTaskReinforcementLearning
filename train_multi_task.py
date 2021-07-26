import os
from copy import deepcopy
import pickle
import numpy as np
import matplotlib.pyplot as plt

from function_approximator import Actor, Critic
from experience_bank import ExperienceBank
from wrapped_env import CartPoleCustom, AcrobotCustom
from utility_functions import *

##############
### Inputs ###
##############

gamma = 0.99                  # Discount factor
n_td_steps = 2                # n TD steps
replay_ratio = 4              # Number of replayed traj for every new traj 
n_episodes_task_switch = 500  # Number of episodes between env switch when training (sequentially)

entropy_coeff = 0.01       # Entropy regularisation for cross entropy loss
policy_clone_coeff = 0.005 # Amount to penalise when diverging from experience policy (during off-policy training)
value_clone_coeff = 0.001  # Amount to penalise when diverging from experience value function

n_episodes = 4000   # Number of episodes to train for
r_avg_window = 100  # Averaging window when calculating score


##############
### Script ###
##############

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

# Create folder to store output
try:
  os.makedirs(os.path.join('multi_task_results'))
except:
  pass
  
# Training loop
r_all = [np.zeros((n_episodes, )), np.zeros((n_episodes, ))]
r_all_avg = [np.nan*np.ones((n_episodes, )), np.nan*np.ones((n_episodes, ))]
for i_episode in range(n_episodes):
   
  i_train = np.mod(int(i_episode/n_episodes_task_switch), 2)
  i_test = i_train - 1
  
  env_train = envs[i_train]
  env_test = envs[i_test]
  
  ### Train
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
    
    # Accumulate gradients after n TD steps
    if np.mod(i_time+1, n_td_steps) == 0:
      n_replay = get_n_replay(replay_ratio) # Int
      i_on_policy = [exp.get_latest_idx(), ] # List
      i_off_policy = list(exp.get_sample_idx(n_replay)) # List
      
      # Sample experience
      for i_exp, exp_idx in enumerate(i_on_policy+i_off_policy):
      
        # Get trajectory from experience bank 
        # When i_exp=0, latest on-policy trajectory will be retrieved
        exp_s, exp_a, exp_r, exp_mu, exp_V, exp_s_prime, exp_terminal, exp_k = exp.get_traj(exp_idx)
        
        Vcurr_prime = func_V.predict_from_lagged(exp_s_prime)[0, 0] # Scalar
        if exp_terminal:
          Vtr = 0
        else:
          pi = func_pi.predict(exp_s_prime) # 1 x n_actions
          Vtr = deepcopy(Vcurr_prime)
          
        # n-step TD (on-policy for i_exp=0, off-policy otherwise)
        for i in reversed(range(exp_k)):
          
          pi = func_pi.predict(exp_s[i:i+1, :]) # 1 x n_actions
          
          # Importance sampling
          rho = pi/exp_mu[i:i+1, :] # 1 x n_actions
          rho_bar = np.min((1, rho[0, exp_a[i, 0]])) # scalar
          Vcurr = func_V.predict_from_lagged(exp_s[i:i+1, :])[0, 0] # scalar
          
          g1 = np.min((10, rho[0, exp_a[i, 0]]))*(exp_r[i, 0] + gamma*Vtr - Vcurr) # Scalar, Vtr is at i+1
          actor_train_y = np.zeros(pi.shape)
          actor_train_y[0, exp_a[i, 0]] += g1
          if i_exp > 0:
            actor_train_y += policy_clone_coeff*exp_mu[i:i+1, :]
          
          func_pi.accumulate_training_data(exp_s[i, :], actor_train_y, entropy_coeff)
          
          # Recursive V-trace targets for c = rho
          Vtr = rho_bar*(exp_r[i, 0] + gamma*Vtr) + (1 - rho_bar)*Vcurr # Scalar, Vtr at i
          
          if i_exp > 0:
            func_V.accumulate_training_data(
              exp_s[i, :], Vtr*np.ones((1, 1)), exp_V[i, 0], value_clone_coeff)
          else:
            # Set Y_clone to Vtr to disable cloning for on-policy
            func_V.accumulate_training_data(
              exp_s[i, :], Vtr*np.ones((1, 1)), Vtr*np.ones((1, 1)), value_clone_coeff) 
          
          Vcurr_prime = exp_r[i, 0] + gamma*Vcurr
    
    # Update total rewards and transition to next step
    r_all[i_train][i_episode] += r
    if done:
      break
    s = s_prime
    
    # End of training episode
  
  ### Test
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
  
  # Averaging results
  if i_episode >= r_avg_window:
    for i_train_test in range(2):
      r_all_avg[i_train_test][i_episode] = np.mean(r_all[i_train_test][i_episode-r_avg_window:i_episode])
    
  if np.mod(i_episode+1, n_episodes_task_switch)==0:
    # Output results periodically to track progress
    print('Episode {}, env_0_r_avg {}, env_1_r_avg {}, training {}'.format(
      i_episode+1, r_all_avg[0][i_episode], r_all_avg[1][i_episode], i_train))
    
    # Save agent and scores at intervals
    with open(os.path.join('multi_task_results', 'agent_{:07d}.p'.format(i_episode+1)), 'wb') as f:
      pickle.dump([func_pi, func_V], f)
    with open(os.path.join('multi_task_results', 'residuals_{:07d}.p'.format(i_episode+1)), 'wb') as f:
      pickle.dump([r_all, r_all_avg], f)
  
  # Save experience bank
  if np.mod(i_episode+1, 2*n_episodes_task_switch)==0:
    with open(os.path.join('multi_task_results', 'exp_bank.p'), 'wb') as f:
      pickle.dump(exp, f)
  
envs[0].close()
envs[1].close()

plt.plot(np.arange(n_episodes), r_all_avg[0], 'b')
plt.plot(np.arange(n_episodes), r_all_avg[1], 'r')
plt.grid()

plt.show()