
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256, 512)
        self.fc512 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 10)
        self.fc128 = nn.Linear(128, 128)
        self.fc256 =nn.Linear(256, 256)
        self.dropout = nn.Dropout(.01)
        self.fc25610 = nn.Linear(256, 10)
        self.fc12810 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        
        for i in range(8):
            x = self.dropout(x)
            x = self.fc128(x)
        x = self.fc2(x)
        # for i in range(4):
        #     x = self.dropout(x)
        #     x = self.fc256(x)
        # x = self.fc3(x)
        # for i in range(8):
        #     x = self.dropout(x)
        #     x = self.fc512(x)
        # x = self.fc4(x)

        for i in range(16):
            x = self.dropout(x)
            x = self.fc256(x)    
        x = self.fc5(x)    
        # x = self.fc6(x)
        # x = self.fc4(x)
        
        for i in range(1):
            x = self.dropout(x)
            x = self.fc128(x)
        x = self.fc12810(x)
        return x

