import torch
import torch.nn as nn
from math import sqrt

class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in  = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.w_q = nn.Linear(dim_in, dim_k, bias=False)
        self.w_k = nn.Linear(dim_in, dim_k, bias=False)
        self.w_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k) # 除以根号d_k

    def forward(self, x, mask=None, k_cache=None, v_cache=None):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        if k_cache is not None and v_cache is not None:
            k = torch.cat([k_cache, k], dim=1) # batch, n_prev + n, dim_k
            v = torch.cat([v_cache, v], dim=1)
        k_cache = k
        v_cache = v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact # q * k的转置 batch, n, n
        
        if mask is not None:
            dist = dist.masked_fill(mask == 0, float('-inf'))
        
        dist = torch.softmax(dist, dim=-1) # batch, n, n
        att = torch.bmm(dist, v) 
        return att, k_cache, v_cache

if __name__ == "__main__":
    batch = 2
    n_x = 4
    d_x = 80
    x = torch.randn(batch, n_x, d_x)
    SA = SelfAttention(dim_in=d_x, dim_k=80, dim_v=64)
    k_cache = None
    v_cache = None

    for t in range(n_x):
        x_t = x[:, t : t+1, :] # batch, 1, d_x
        mask = torch.tril(torch.ones(1, t + 1)).unsqueeze(0).repeat(batch, 1, 1)
        att, k_cache, v_cache = SA(x_t, mask=mask, k_cache=k_cache, v_cache=v_cache)
        print(f"时间步{t + 1}: 输出形状 {att.shape}")