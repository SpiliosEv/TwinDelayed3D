import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class ActorNetwork(nn.Module):
  def __init__(self, alpha , input_dims, fc1_dims, fc2_dims, n_actions):
    super(ActorNetwork, self).__init__()
    self.input_dims = input_dims
    self.n_actions = n_actions
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims 

    self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
    self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    self.mu = nn.Linear(self.fc2_dims, self.n_actions)

    self.optimizer = optim.Adam(self.parameters(), lr=alpha)
    self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    self.to(self.device)

  def forward(self, state):
    prob = self.fc1(state)
    prob = F.relu(prob)
    prob = self.fc2(prob)
    prob = F.relu(prob)
    prob = self.mu(prob)
    prob = torch.tensor(1/np.sqrt(2))*torch.tanh(prob)

    return prob

