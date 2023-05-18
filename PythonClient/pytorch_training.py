import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transform

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)


class MagicalBrain(nn.Module):
    
    def __init__(self, inpust_size, ) -> None:
        super(MagicalBrain, self).__init__()
        self.fc1 = nn.Linear(inpust_size, 100)
        self.fc2 = nn.Linear(100, 20) 
        self.fc3 = nn.Linear(20, 1)
        self.relu = nn.ReLU()
        super().to(device=device)
   
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x