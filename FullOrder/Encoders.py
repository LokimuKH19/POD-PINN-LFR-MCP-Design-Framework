import torch.nn as nn
import torch.nn.functional as F


# Encoding the Coordinate
class CoordMLP(nn.Module):
    def __init__(self, hidden_dim=32, input_features=3, output_features=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_features),
            nn.Tanh()
        )

    def forward(self, coords):
        """
        coords: (B, N, input_features)
        return: (B, N, output_features)
        """
        B, N, C = coords.shape
        x = coords.view(B * N, C)
        out = self.net(x)
        out = out.view(B, N, -1)
        return out


# New CoordEncoder
class CoordCNN(nn.Module):
    def __init__(self, input_features=3, output_features=32, hidden_dim=64):
        super().__init__()

        # PointNet-style 1D conv layers
        self.conv1 = nn.Conv1d(input_features, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv3 = nn.Conv1d(hidden_dim, output_features, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(output_features)

    def forward(self, coords):
        """
        coords: (B, N, 3)
        returns: (B, N, output_features)
        """
        # Permute for Conv1d: (B, C_in, N)
        x = coords.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Back to (B, N, output_features)
        x = x.permute(0, 2, 1)
        return x