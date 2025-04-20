import numpy as np

class Linear():
    def __init__(self, dim_in, dim_out):
        '''
        线性全连接层
        dim_in: 输入维度
        dim_out: 输出维度
        '''
        scale = np.sqrt(dim_in / 2)
        self.weight = np.random.standard_normal((dim_in, dim_out)) / scale
        self.bias = np.zeros((1, dim_out))
        self.params = [self.weight, self.bias]

    def __call__(self, x):
        '''
        x: 输入 (batch_size, dim_in)
        '''
        return self.forward(x)
    
    def forward(self, x):
        '''
        x: 输入 (batch_size, dim_in)
        '''
        self.x = x
        return np.dot(self.x, self.weight) + self.bias
    
    def backward(self, d_out):
        '''
        d_out: 输出的梯度 (batch_size, dim_out)
        '''
        d_x = np.dot(d_out, self.weight.T)
        d_w = np.dot(self.x.T, d_out)
        d_b = np.mean(d_out, axis=0)

        return d_x, [d_w, d_b]
    
if __name__ == "__main__":
    x = np.random.randn(2, 10)
    fc = Linear(10, 2)
    out = fc(x)
    print(out.shape)
    d_out = np.random.randn(2, 2)
    d_x, grads = fc.backward(d_out)
    print(d_x.shape, grads[0].shape, grads[1].shape)
    print(d_x, grads[0], grads[1])