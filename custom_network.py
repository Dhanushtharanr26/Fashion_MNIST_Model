import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionNet(nn.Module):

    def __init__(self):
        super(FashionNet, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(784, 128)

        # Branch A
        self.a1 = nn.Linear(128, 64)
        self.a2 = nn.Linear(64, 64)

        # Branch B
        self.b1 = nn.Linear(128, 64)
        self.b2 = nn.Linear(64, 64)

        # Output
        self.output = nn.Linear(128, 10)

    def forward(self, x):

        x = self.flatten(x)

        x = F.relu(self.fc1(x))

        # Branch A
        skip = x[:, :64]

        a = F.relu(self.a1(x))
        a = self.a2(a)

        a = a + skip
        a = F.relu(a)

        # Branch B
        b = F.relu(self.b1(x))
        b = F.relu(self.b2(b))

        # Concatenate
        merged = torch.cat((a, b), dim=1)

        out = self.output(merged)

        return out