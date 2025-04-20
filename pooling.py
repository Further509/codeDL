import numpy as np

def max_pooling(inputs, pool_size, stride):
    '''
    input: (C, H, W)
    pool_size: 池化核大小
    stride: 步长
    '''
    C, H, W = inputs.shape

    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    outputs = np.zeros((C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            inputs_slice = inputs[:, i * stride: i * stride + pool_size, j * stride: j * stride + pool_size]
            outputs[:, i, j] = np.max(inputs_slice, axis=(1, 2))
    
    return outputs

def average_pooling(inputs, pool_size, stride):
    '''
    input: (C, H, W)
    pool_size: 池化核大小
    stride: 步长
    '''
    C, H, W = inputs.shape

    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    outputs = np.zeros((C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            inputs_slice = inputs[:, i * stride: i * stride + pool_size, j * stride: j * stride + pool_size]
            outputs[:, i, j] = np.mean(inputs_slice, axis=(1, 2))
    
    return outputs

if __name__ == "__main__":
    x = np.random.rand(3, 255, 255)
    output = max_pooling(x, 5, 1)
    print(output.shape)

    output = average_pooling(x, 5, 1)
    print(output.shape)