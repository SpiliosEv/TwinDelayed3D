import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
  def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions):
    super(CriticNetwork, self).__init__()
    
    self.input_dims = input_dims 
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.n_actions = n_actions

    self.fc1 = nn.Linear(self.input_dims + self.n_actions, self.fc1_dims)
    self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    self.q = nn.Linear(self.fc2_dims, 1)
     
    self.optimizer = optim.Adam(self.parameters(), lr = beta)
    self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    self.to(self.device)


  def forward(self, state, action):
    q1_action_value = self.fc1(torch.cat([state,action], dim=1))
    q1_action_value = F.relu(q1_action_value)
    q1_action_value = self.fc2(q1_action_value)
    q1_action_value = F.relu(q1_action_value)
    q1_action_value = self.q(q1_action_value)

    return q1_action_value

