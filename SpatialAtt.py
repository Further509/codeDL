import torch
import torch.nn as nn

# 通道压缩为1
class SpatialAttention(nn.Module):
    def __init__(self, kernal_size=7):
        super(SpatialAttention, self).__init__()
        assert kernal_size in (3, 5, 7), 'kernel size must be 3, 5 or 7'
        padding = 3 if kernal_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernal_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        avg_out = torch.mean(x, dim=1, keepdim=True) # [batch_size, 1, height, width]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [batch_size, 1, height, width]
        x = torch.cat([avg_out, max_out], dim=1) # [batch_size, 2, height, width]
        x = self.conv1(x) # [batch_size, 1, height, width]
        return self.sigmoid(x) 
    
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)  # Batch size of 2, 64 channels, 32x32 feature map
    spatial_attention = SpatialAttention(kernal_size=7)
    output = spatial_attention(x)
    print(output.shape)  # (2, 1, 32, 32)