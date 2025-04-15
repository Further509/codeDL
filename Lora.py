import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, merge, rank=16, lora_alpha=16, dropout=0.5):
        super(LoraLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout

        self.linear = nn.Linear(in_features, out_features)
        if self.rank > 0:
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
            self.scale = self.lora_alpha / self.rank
            self.linear.weight.requires_grad = False # 冻结参数

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()
        
        self.initial_weights()

    def initial_weights(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        if self.rank > 0 and self.merge:
            output = F.linear(x, self.linear.weight + self.lora_b @ self.lora_a * self.scale, self.linear.bias)
            output = self.dropout(output)
            return output
        else:
            return self.dropout(self.linear(x))
        
if __name__ == "__main__":
    batch = 2
    n_x = 4
    d_x = 80
    x = torch.randn(batch, n_x, d_x)
    # W * x + b 
    lora = LoraLinear(in_features=d_x, out_features=4, merge=True, rank=16, lora_alpha=16, dropout=0.5)
    output = lora(x)
    print(output.shape)