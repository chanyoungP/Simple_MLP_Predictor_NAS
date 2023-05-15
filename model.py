'''MLP MODEL AS a predictor '''
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(49,64)
        self.fc3 = nn.Linear(35,64)
        self.fc4 = nn.Linear(7,64)

        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 32)
        self.fc10 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self,x1,x2,x3,x4):
        x1 = self.fc1(x1)
        x1 = self.relu(x1)

        x2 = self.fc2(x2)
        x2 = self.relu(x2)

        x3 = self.fc3(x3)
        x3 = self.relu(x3)

        x4 = self.fc4(x4)
        x4 = self.relu(x4)

        x = torch.concat((x1,x2,x3,x4),dim=1)

        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.fc10(x)
        return x
