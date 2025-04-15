import torch
import torch.nn as nn
from math import sqrt

# 实际上就是增加一个维度再变回
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        '''
        dim_in: 输入维度
        dim_k: query和key的维度
        dim_v: value的维度
        num_heads: 多头头数 
        '''
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.w_q = nn.Linear(dim_in, dim_k, bias=False)
        self.w_k = nn.Linear(dim_in, dim_k, bias=False)
        self.w_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x, mask=None):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh
        dv = self.dim_v // nh

        q = self.w_q(x).reshape(batch, n, nh, dk).transpose(1, 2) # batch, nh, n, dk
        k = self.w_k(x).reshape(batch, n, nh, dk).transpose(1, 2) # batch, nh, n, dk
        v = self.w_v(x).reshape(batch, n, nh, dv).transpose(1, 2) # batch, nh, n, dv

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact # batch, nh, n, n
        
        if mask is not None:
            dist = dist.masked_fill(mask == 0, float('-inf'))
        
        dist = torch.softmax(dist, dim=-1) # batch, nh, n, n

        att = torch.matmul(dist, v) # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v) # dim_v = nh * dv
        return att

if __name__ == "__main__":
    batch = 2
    n_x = 4
    d_x = 80
    x = torch.randn(batch, n_x, d_x)
    MHA = MultiHeadSelfAttention(dim_in=d_x, dim_k=160, dim_v=64, num_heads=4)

    mask = torch.tril(torch.ones(n_x, n_x)).unsqueeze(0).unsqueeze(0)
    print(mask.shape)
    att = MHA(x, mask=mask)
    print(att.size())