import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical

from models import ValueNetwork, PolicyNetwork


class A2CAgent():

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
    
    def compute_loss(self, trajectory, adv_method):
        """   
        When gamma is large, the NN loss does not converge, we should use MC to estimate advantage. 
        When gamma is small (i.e. 0.9), the NN loss decreases after training, we can use TD to estimate advantage. 
        """
        states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        actions = torch.LongTensor([sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
        next_states = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
        dones = torch.FloatTensor([sars[4] for sars in trajectory]).view(-1, 1).to(self.device)
        
        # compute value target
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))])\
             * rewards[j:]) for j in range(rewards.size(0))]  # sorry, not the most readable code.
        value_targets = torch.FloatTensor(discounted_rewards).view(-1, 1)

        # compute value loss
        values = self.value_network.forward(states)
        value_loss = F.mse_loss(values, value_targets.detach())
        
        
        # compute policy loss with entropy bonus
        logits = self.policy_network.forward(states)
        dists = logits
        probs = Categorical(dists)
        
        # compute entropy bonus
        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torch.stack(entropy).sum()
      
        # 0 for MC, 1 for TD
        if adv_method == 0:
            advantages = value_targets - values
        if adv_method == 1:
            advantages = rewards - values + self.gamma * torch.cat((values[1:], torch.FloatTensor([[0]])), dim = 0)
        
        
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantages.detach()
        policy_loss = policy_loss.sum() - 0.001 * entropy

        return value_loss, policy_loss
    
    def update(self, trajectory, adv_method):
        value_loss, policy_loss = self.compute_loss(trajectory, adv_method)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
