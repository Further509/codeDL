import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELosswithLogits(nn.Module):
    def __init__(self, posweight=1, reduction='mean'):
        super(BCELosswithLogits, self).__init__()
        self.posweight = posweight
        self.reduction = reduction

    def forward(self, logits, targets):
        ligits = F.sigmoid(logits)
        loss = - self.posweight * targets * torch.log(logits) - \
            (1 - targets) * torch.log(1 - logits)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum() 
        
class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [N, C, H, W], targets: [N, H, W]
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)
            logits = logits.transpose(1, 2)
            logits = logits.contiguous().view(-1, logits.size(2))
        targets = targets.view(-1)

        logits = F.log_softmax(logits, dim=1)
        logits = logits.gather(1, targets.unsqueeze(1))
        loss = -1 * logits
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

if __name__ == "__main__":
    # logits = torch.tensor([0.5, 0.2, 0.8], requires_grad=True)
    logits = torch.tensor([0.2, 0.1, 0.8], requires_grad=True)
    targets = torch.tensor([1, 0, 1], dtype=torch.float32)
    criterion = BCELosswithLogits(posweight=2)
    loss = criterion(logits, targets)
    print(loss.item())

    logits = torch.tensor([[0.5, 0.2, 0.8], [0.1, 0.9, 0.4]], requires_grad=True)
    targets = torch.tensor([[2, 1]], dtype=torch.long)
    criterion = CrossEntropyLoss()
    loss = criterion(logits, targets)
    print(loss.item())