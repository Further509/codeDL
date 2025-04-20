import numpy as np

class Dropout():
    def __init__(self, p):
        '''
        p: 保留比例
        '''
        self.p = p

    def __call__(self, x, mode):
        '''
        mode: train or test
        '''
        return self.forward(x, mode)
    
    def forward(self, x, mode):
        if mode == 'train':
            self.mask = np.random.binomial(1, self.p, x.shape) / self.p
            out = self.mask * x
        else:
            out = x
        return out
    
    def backward(self, d_out):
        '''
        d_out: loss对dropout输出的梯度
        '''
        return d_out * self.mask
    
if __name__ == "__main__":
    x = np.random.randn(2, 10)
    dropout = Dropout(0.5)
    out = dropout(x, "train")
    print(out.shape)
    d_out = np.random.randn(2, 10)
    grads = dropout.backward(d_out)
    print(grads.shape)
    print(grads)