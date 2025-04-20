import numpy as np

class ReLu(object):
    def __init__(self):
        self.X = None

    def __call__(self, X):
        self.X = X
        return self.forward(X)
    
    def forward(self, X):
        return np.maximum(0, X)
    
    def backward(self, grad_output):
        '''
        grad_output: loss对relu激活输出的梯度
        '''
        grad_relu = self.X > 0
        return grad_relu * grad_output
    
if __name__ == "__main__":
    relu = ReLu()
    x = np.random.randn(2, 10)
    print("x: ", x)
    out = relu(x)
    print("out: ", out)
    d_out = np.random.randn(2, 10)
    grads = relu.backward(d_out)
    print("grad: ", grads)