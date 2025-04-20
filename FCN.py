import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc2(self.relu(self.fc1(x)))
        return out
    
if __name__ == "__main__":
    x = torch.randn(1, 10)
    model = FCN(10, 5, 2)
    print(model(x).shape)