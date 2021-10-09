import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from Replay_Memory import ReplayBuffer
from CriticNetwork import CriticNetwork
from Actor import ActorNetwork

class Agent(object):
  def __init__(self, alpha = 0.0002, beta = 0.0002, input_dims = 3, tau = 0.05, gamma = 0.99, update_actor_interval = 2, n_actions=3, max_size=1000000,
               layer1_size = 200, layer2_size = 200, batch_size = 100, noise=0.1):
    
    self.gamma = gamma
    self.tau = tau
    self.memory = ReplayBuffer(max_size, input_dims, n_actions)
    self.batch_size = batch_size
    self.n_actions = n_actions
    self.update_actor_iter = update_actor_interval
    self.learn_step_cntr = 0
    
    self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions)
    self.critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions = n_actions)
    self.critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions = n_actions)

    self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions)
    self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions = n_actions)
    self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions = n_actions)

    self.noise = noise
    self.update_network_parameters(tau=1)

  def choose_action(self, observation):
    state = torch.tensor([observation]).to(self.actor.device)
    mu = self.actor.forward(state).to(self.actor.device)
    mu_prime = mu +  torch.tensor(np.random.normal(scale=self.noise), dtype = torch.float).to(self.actor.device)
    mu_prime = torch.clamp(mu_prime, -torch.tensor(1/np.sqrt(2)), torch.tensor(1/np.sqrt(2)))
    

    return mu_prime.cpu().detach().numpy()[0]

  
  def remember(self, state, action, reward, new_state):
    self.memory.store_transition(state, action, reward, new_state)

  def learn(self):
    if self.memory.mem_cntr < self.batch_size:
      return
    state, action, reward, new_state = self.memory.sample_buffer(self.batch_size)

    reward = torch.tensor(reward, dtype = torch.float).to(self.critic_1.device)
    state_ = torch.tensor(new_state, dtype = torch.float).to(self.critic_1.device)
    state = torch.tensor(state, dtype = torch.float).to(self.critic_1.device)
    action = torch.tensor(action, dtype = torch.float).to(self.critic_1.device)
    target_actions = self.target_actor.forward(state_)
    target_actions = target_actions +  torch.clamp(torch.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
    target_actions = torch.clamp(target_actions, -torch.tensor(1/np.sqrt(2)), torch.tensor(1/np.sqrt(2)))

    q1_ = self.target_critic_1.forward(state_, target_actions)
    q2_ = self.target_critic_2.forward(state_, target_actions)

    q1 = self.critic_1.forward(state, action)
    q2 = self.critic_2.forward(state, action)

    q1_ = q1_.view(-1)
    q2_ = q2_.view(-1)

    critic_value_ = torch.min(q1_, q2_)
    target  = reward + self.gamma*critic_value_
    target = target.view(self.batch_size, 1)

    self.critic_1.optimizer.zero_grad()
    self.critic_2.optimizer.zero_grad()

    q1_loss = F.mse_loss(target, q1)
    q2_loss = F.mse_loss(target, q2)
    critic_loss = q1_loss + q2_loss
    critic_loss.backward()
    
    self.critic_1.optimizer.step()
    self.critic_2.optimizer.step()

    self.learn_step_cntr += 1
    if self.learn_step_cntr % self.update_actor_iter != 0:
      return 
    
    self.actor.optimizer.zero_grad()
    actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
    actor_loss = -torch.mean(actor_q1_loss)
    actor_loss.backward()
    self.actor.optimizer.step()

    self.update_network_parameters()

  def update_network_parameters(self, tau=None):
    if tau is None:
      tau = self.tau

    actor_params = self.actor.named_parameters()
    critic_1_params = self.critic_1.named_parameters()
    critic_2_params = self.critic_2.named_parameters()
    target_actor_params = self.target_actor.named_parameters()
    target_critic_1_params = self.target_critic_1.named_parameters()
    target_critic_2_params = self.target_critic_2.named_parameters()

    critic_1 = dict(critic_1_params)
    critic_2 = dict(critic_2_params)
    actor = dict(actor_params)
    target_actor = dict(target_actor_params)
    target_critic_1 = dict(target_critic_1_params)
    target_critic_2 = dict(target_critic_2_params)

    for name in critic_1:
      critic_1[name] = tau*critic_1[name].clone() + (1-tau)*target_critic_1[name].clone()


    for name in critic_2:
      critic_2[name] = tau*critic_2[name].clone() + (1-tau)*target_critic_2[name].clone()

    for name in actor:
      actor[name] = tau*actor[name].clone() + (1-tau)*target_actor[name].clone()
    
    self.target_critic_1.load_state_dict(critic_1)
    self.target_critic_2.load_state_dict(critic_2)

    self.target_actor.load_state_dict(actor)




    








