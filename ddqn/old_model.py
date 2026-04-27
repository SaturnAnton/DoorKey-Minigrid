import torch
import torch.nn as nn
import numpy as np
import random
from collections import namedtuple

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer():
  def __init__(self, num_actions, memory_len = 10000):
      self.memory_len = memory_len
      self.transition = []
      self.num_actions = num_actions

  def add(self, state, action, reward, next_state, done):
      if self.length() > self.memory_len:
        self.remove()
      self.transition.append(Transition(state, action, reward, next_state, done))

  def sample_batch(self,device, batch_size = 32):
      minibatch = random.sample(self.transition, batch_size)
      states_mb, a_, reward_mb, next_states_mb, done_mb = map(np.array, zip(*minibatch))

      mb_reward = torch.from_numpy(reward_mb).to(device=device, dtype=torch.float32)
      mb_done = torch.from_numpy(done_mb.astype(int)).to(device=device)
      a_ = a_.astype(int)
      a_mb = np.zeros((a_.size, self.num_actions), dtype=np.float32)
      a_mb[np.arange(a_.size), a_] = 1
      mb_a = torch.from_numpy(a_mb).to(device=device)
      
      return states_mb, mb_a, mb_reward, next_states_mb, mb_done

  def length(self):
      return len(self.transition)

  def remove(self):
      self.transition.pop(0)

class MlpMinigridPolicy(nn.Module):
    def __init__(self,input_shape, num_actions=7):
        super().__init__()
        self.num_actions = num_actions
        flat_size = input_shape[0] * input_shape[1] * input_shape[2]

        self.fc = nn.Sequential(
           nn.Flatten(),
           nn.Linear(flat_size, 256),
           nn.ReLU(),
           nn.Linear(256, 256),
           nn.ReLU(),
           nn.Linear(256, 64),
           nn.ReLU(),
           nn.Linear(64, num_actions)
        )

    def forward(self, x):
        if len(x.size()) == 3:
          x = x.unsqueeze(dim=0)
        x = x / 10.0
        return self.fc(x)