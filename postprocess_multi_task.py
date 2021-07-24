import os
from copy import deepcopy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio

from function_approximator import Actor, Critic
from experience_bank import ExperienceBank
from wrapped_env import CartPoleCustom, AcrobotCustom

n_episodes = 4000

try:
  os.makedirs(os.path.join('multi_task_results', 'gifs'))
except:
  pass
  
with open(os.path.join('multi_task_results', 'agent_{:07d}.p'.format(n_episodes)), 'rb') as f:
  agent = pickle.load(f)
env = [CartPoleCustom(), AcrobotCustom()]
env_frames = []
for i_env in range(2):
  env_frames.append([])
  for i_test in range(5):
    env_frames[i_env].append([])
    s = env[i_env].reset()
    r_all = 0
    gif_name = os.path.join('multi_task_results', 'gifs', 'env{:d}_test{:d}.gif'.format(i_env, i_test))
    with imageio.get_writer(gif_name, mode='I', duration=0.02) as writer:
      for i_step in range(200):
        pi = agent[0].predict(s[np.newaxis])
        a = np.random.choice(env[i_env].get_n_actions(), 1, p=pi[0, :])[0]
        s_prime, r, done, info = env[i_env].step(a)
        im = env[i_env]._env.render(mode="rgb_array")
        writer.append_data(im)
        r_all += r
        
        if done:
          break
        s = s_prime
      print('env: {}, r: {}'.format(i_env, r_all))
      env[i_env].close()

with open(os.path.join('multi_task_results', 'residuals_{:07d}.p'.format(n_episodes)), 'rb') as f:
  res, res_avg = pickle.load(f)

labs = ('Cartpole', 'Acrobot')
plot_vec = lambda v, lab: plt.plot(np.arange(len(v)), v, label=lab)
plt.figure()
for i in range(2):
  plot_vec(res_avg[i], labs[i])
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()
