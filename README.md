# Multi-Task Reinforcement Learning
Multi-task reinforcement learning using CLEAR algorithm from \[1\] ([link](https://arxiv.org/abs/1811.11682)). Actor-critic reinforcement learning used to learn 2 tasks in a single policy without demarcating tasks when learning. Off-policy learning and behavioural cloning from experience replay was introduced to prevent catastrophic forgetting.

Dependecies include gym, numpy, matplotlib and imageio. Reinforcement learning agent and machine learning classes written entirely in numpy for learning purposes.

## File descriptors
* <code>copy_cartpole_custom.py</code>
  * Copy custom cartpole class (which has 3 discrete action space instead of 2) to installed gym location.
* <code>train_single_task.py</code>
  * Trains an agent on single cartpole task.
* <code>postprocess_single_task.py</code>
  * Post process results from trained agent to obtain gif.
* <code>train_multi_task.py</code>
  * Trains an agent on both cartpole and acrobot case.
* <code>postprocess_multi_task.py</code>
  * Post process results from trained agent to obtain gif.

First run <code>copy_cartpole_custom.py</code>, then followed by either of the training script and the respective post process script. Single task takes about 15 minutes to train while multi task takes couple of hours. The code was written for transparency, not speed.

## Results
<img src='sample_img/training_sample.png' width=600>

<img src='sample_img/cartpole_sample.gif' height=250> <img src='sample_img/acrobot_sample.gif' height=250>

## References
\[1\] D. Rolnick et al., "Experience Replay for Continual Learning", in NeurIPS, 2019.