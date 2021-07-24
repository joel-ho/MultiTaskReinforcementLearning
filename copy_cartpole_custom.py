import os
import shutil
import gym

custom_env_name = 'cartpole_custom.py'
gym_folder = os.path.abspath(os.path.join(gym.__file__, '..'))
shutil.copyfile(
  custom_env_name, 
  os.path.join(gym_folder, 'envs', 'classic_control', custom_env_name))