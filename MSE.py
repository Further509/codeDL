import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        loss = (input - target) ** 2
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        
if __name__ == "__main__":
    input = torch.tensor([0.5, 0.2, 0.8], requires_grad=True)
    target = torch.tensor([0.1, 0.4, 0.9])
    criterion = MSELoss(reduction='mean')
    loss = criterion(input, target)
    print(loss.item())