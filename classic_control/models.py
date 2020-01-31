import torch
import torch.nn as nn
import torch.nn.functional as F 

class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 8)
        self.fc3 = nn.Linear(8, output_dim)

    def forward(self, state):
        value = F.relu(self.fc1(state))
        value = self.fc2(value)
        value = self.fc3(value)

        return value
    

class PolicyNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
    
    def forward(self, state):
        logits = F.relu(self.fc1(state))
        logits = F.softmax(self.fc2(logits), dim = 0)

        return logits
