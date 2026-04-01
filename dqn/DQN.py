import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network con due hidden layer.
    L'aggiunta del secondo layer migliora la capacità rappresentativa
    necessaria per ambienti complessi come DoorKey-8x8.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)   # ← secondo hidden layer aggiunto
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))                        # ← secondo hidden layer aggiunto
        return self.fc3(x)


if __name__ == '__main__':
    # Test rapido
    state_dim  = 193   # 8*8*3 + 1 con FullyObsWrapper
    action_dim = 7
    net   = DQN(state_dim, action_dim)
    state = torch.randn(1, state_dim)
    out   = net(state)
    print(f"Output shape: {out.shape}")   # atteso: torch.Size([1, 7])
    print(out)