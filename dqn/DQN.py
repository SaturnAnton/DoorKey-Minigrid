import torch
from torch import nn
import torch.nn.functional as F

#vado a creare una classe per definire la DQN
class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        #Qui vado a definire i layer (in questo caso ci sono 2 hidden layer)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    #permette la parte di calcolo
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))                        
        return self.fc3(x)


if __name__ == '__main__':
    state_dim  = 76
    action_dim = 7
    net   = DQN(state_dim, action_dim)
    state = torch.randn(1, state_dim)
    out   = net(state)
    print(f"Output shape: {out.shape}")
    print(out)