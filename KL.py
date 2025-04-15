import torch
import torch.nn.functional as F

# 计算KL散度
def DKL(_p, _q):
    return torch.sum(_p * torch.log(_p / _q), dim=-1)

if __name__ == "__main__":
    p = torch.tensor([0.4, 0.6])
    q = torch.tensor([0.3, 0.7])
    divergence = DKL(p, q)
    print(divergence) 
    print(F.kl_div(q.log(), p, reduction='sum'))
