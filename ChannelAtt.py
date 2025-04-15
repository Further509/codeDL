import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # batch, C, H, W
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x)))) 
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)  # Batch size of 2, 64 channels, 32x32 feature map
    channel_attention = ChannelAttention(in_planes=64)
    output = channel_attention(x)
    print(output.shape)  # Should be (2, 64, 1, 1)