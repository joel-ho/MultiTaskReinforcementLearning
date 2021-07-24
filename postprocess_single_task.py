import os
from copy import deepcopy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio
import gym

from function_approximator import Actor, Critic
from experience_bank import ExperienceBank
from wrapped_env import CartPoleCustom, AcrobotCustom

n_episodes = 1500

try:
  os.makedirs(os.path.join('single_task_results', 'gifs'))
except:
  pass

with open(os.path.join('single_task_results', 'agent_{:07d}.p'.format(n_episodes)), 'rb') as f:
  agent = pickle.load(f)
env = gym.make('CartPole-v0')

for i_test in range(5):
  s = env.reset()
  r_all = 0
  gif_name = os.path.join('single_task_results', 'gifs', 'test{:d}.gif'.format(i_test))
  with imageio.get_writer(gif_name, mode='I', duration=0.02) as writer:
    for i_step in range(200):
      pi = agent[0].predict(s[np.newaxis])
      a = np.random.choice(env.action_space.n, 1, p=pi[0, :])[0]
      s_prime, r, done, info = env.step(a)
      im = env.render(mode="rgb_array")
      writer.append_data(im)
      r_all += r
      
      if done:
        break
      s = s_prime
    print('r: {}'.format(r_all))
    env.close()
