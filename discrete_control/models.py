import torch
import torch.nn as nn
import torch.nn.functional as F 

class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, state):
        value = torch.tanh(self.fc1(state))
        value = torch.tanh(self.fc2(value))
        value = self.fc3(value)

        return value
    
class PolicyNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, state):
        logits = torch.tanh(self.fc1(state))
        logits = torch.tanh(self.fc2(logits))
        logits = F.softmax(self.fc3(logits), dim = 0)
        return logits
        