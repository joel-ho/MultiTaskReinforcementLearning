from copy import deepcopy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gym

from function_approximator import Actor, Critic
from experience_bank import ExperienceBank

# Settings
gamma = 0.99
n_td_steps = 2
replay_ratio = 4

entropy_coeff = 0.005
policy_clone_coeff = 0.01
value_clone_coeff = 0.005

n_episodes = 1500
r_avg_window = 50
save_agent_interval = 500

# Environment
env = gym.make('CartPole-v0')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# Actor
func_pi = Actor(
    (n_states, 64, 64, n_actions), 
    ('relu', 'relu', 'softmax_cross_entropy'), 
    'cross_entropy_entropy_reg', 
    'adam', 
    32)

# Critic
func_V = Critic(
    (n_states, 64, 64, 1), 
    ('relu', 'relu', 'identity'), 
    'mean_squared_error_clone', 
    'adam', 
    32,
    use_lag=True)

# Experience
exp = ExperienceBank(int(1e6), n_td_steps, n_states, n_actions)

# Poisson distr
def poisson(lam, minimum=1, maximum=8):
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
r_all = np.zeros((n_episodes, ))
r_all_avg = np.nan*np.ones((n_episodes, ))
for i_episode in range(n_episodes):

  # Initialize state
  s = env.reset()

  # Sample trajectory
  for i_time in range(200):
    
    # Perform action
    pi = func_pi.predict(s[np.newaxis]) # 1 x n_actions
    V = func_V.predict(s[np.newaxis])[0, 0] # Scalar
    a = np.random.choice(n_actions, 1, p=pi[0, :])[0]
    s_prime, r, done, info = env.step(a)
    
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
    r_all[i_episode] += r
    if done:
      break
    s = s_prime
    
    # End of episode
  
  if i_episode >= r_avg_window:
    r_all_avg[i_episode] = np.mean(r_all[i_episode-r_avg_window:i_episode])
    
  if np.mod(i_episode+1, 100)==0:
    print('Episode {}, r_avg {}'.format(i_episode+1, r_all_avg[i_episode]))
  if np.mod(i_episode+1, save_agent_interval)==0:
    with open(('single_task_data\\agent_%07d.p'%(i_episode+1)), 'wb') as f:
      pickle.dump([func_pi, func_V], f)
  
env.close()

plt.plot(np.arange(n_episodes), r_all, 'b')
plt.plot(np.arange(n_episodes), r_all_avg, 'r')
plt.grid()

plt.show()