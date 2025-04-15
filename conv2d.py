import numpy as np

def conv2d(img, in_channel, out_channels, kernels, bias, stride=1, padding=0):
    '''
    实现2D卷积操作

    参数:
    - img: 输入图像，形状为 (N, C, H, W)
    - in_channels: 输入通道数
    - out_channels: 输出通道数
    - kernels: 卷积核，形状为 (kh, kw)
    - bias: 偏置，形状为 (N, out_channels)
    - stride: 步幅
    - padding: 填充大小

    返回:
    - outputs: 卷积结果，形状为 (N, out_channels, out_h, out_w)
    '''
    N, C, H, W = img.shape
    kh, kw = kernels.shape
    p = padding
    assert C == in_channels, "kernels 输入通道不匹配"
    # 边界填充padding
    if p:
        img = np.pad(img, ((0, 0),(0, 0),(p, p),(p, p)), 'constant') # 前两维度不填充
    
    out_h = (H + 2 * p - kh) // stride + 1
    out_w = (W + 2 * p - kw) // stride + 1

    outputs = np.zeros((N, out_channels, out_h, out_w))

    for n in range(N):
        for out in range(out_channels):
            for i in range(in_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        for x in range(kh):
                            for y in range(kw):
                                outputs[n][out][h][w] += img[n][i][h * stride + x][w * stride + y]
                if i == in_channels - 1:
                    outputs[n][out][:][:] += bias[n][out]
    
    return outputs

if __name__ == "__main__":
    batch = 2
    in_channels = 3
    out_channels = 2
    height, width = 5, 5
    kernel_size = 3
    img = np.random.randn(batch, in_channels, height, width)
    kernels = np.random.randn(kernel_size, kernel_size)
    bias = np.random.randn(batch, out_channels)

    outputs = conv2d(img, in_channels, out_channels, kernels, bias, stride=1, padding=1)
    print(outputs.shape)