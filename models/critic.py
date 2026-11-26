import torch
import torch.nn as nn   

class Single_Head_Critic(nn.Module):
    def __init__(self, global_state_dim, hidden_dim=64):
        super(Single_Head_Critic, self).__init__()
        
        self.fc1 = nn.Linear(global_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) 

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, global_state):
        x = torch.relu(self.fc1(global_state))
        x = torch.relu(self.fc2(x))
        
        val_head = self.head(x)
        
        return val_head

class Two_Head_Critic(nn.Module):
    def __init__(self, global_state_dim, hidden_dim=64):
        super(Two_Head_Critic, self).__init__()

        self.fc1 = nn.Linear(global_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.seeker = nn.Linear(hidden_dim, 1)
        self.guard = nn.Linear(hidden_dim, 1)

    def forward(self, global_state):
        x = torch.relu(self.fc1(global_state))
        x = torch.relu(self.fc2(x))

        val_seeker = self.seeker(x)
        val_guard = self.guard(x)

        return val_seeker, val_guard