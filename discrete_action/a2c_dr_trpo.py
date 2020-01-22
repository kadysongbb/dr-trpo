import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical

from models import ValueNetwork, PolicyNetwork


class DRTRPOAgent1():

    def __init__(self, env, gamma, lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.gamma = gamma
        self.lr = lr
        
        self.value_network = ValueNetwork(self.obs_dim, 1)
        self.policy_network = PolicyNetwork(self.obs_dim, self.action_dim)
        
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        logits = self.policy_network.forward(state)
        dist = logits
        probs = Categorical(dist)
        return probs.sample().cpu().detach().item()

    def compute_adv_first_sa(self, trajectory):
        """
        Compute the advantage of the first state, action pair in the trajectory.
        The advantage is estimated using MC: i.e. discounted reward sum (from trajectory) - value (from NN)
        """
        states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        actions = torch.LongTensor([sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
        next_states = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
        dones = torch.FloatTensor([sars[4] for sars in trajectory]).view(-1, 1).to(self.device)

        # compute value target
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))])\
             * rewards[j:]) for j in range(rewards.size(0))]  
        value_targets = torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)

        # compute value loss
        values = self.value_network.forward(states)
        value_loss = F.mse_loss(values, value_targets.detach())

        advantages = value_targets - values
        return advantages[0], value_loss

    def compute_adv(self, state, next_state, reward):
        """
        Compute the advantage using TD method: i.e. r + v(s') - v(s) - depends highly on the accuracy of NN
        """
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.as_tensor(reward)
        state_value = self.value_network.forward(state)
        next_state_value = self.value_network.forward(next_state)
        value_target = reward + next_state_value 
        advantage = value_target - state_value
        value_loss = F.mse_loss(state_value, value_target)
        return advantage, value_loss
    
    def update(self, value_loss, policy_loss):
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


    def compute_policy_loss(self, state, state_adv):
        beta = 1
        state = torch.FloatTensor(state).to(self.device)
        logits = self.policy_network.forward(state)
        pi_dist = logits
        state_adv = torch.FloatTensor(state_adv).to(self.device)
        denom = torch.sum(torch.exp(state_adv/beta)*pi_dist)
        new_pi_dist = torch.exp(state_adv/beta)*pi_dist/denom
        return F.mse_loss(pi_dist, new_pi_dist)
