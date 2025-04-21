import numpy as np

class Tanh():
    def __init__(self):
        self.X = None

    def __call__(self, X):
        self.X = X
        return self.forward(self.X)
    
    def forward(self, X):
        return np.tanh(X)
    
    def backward(self, grad_output):
        grad_tanh = 1 - (np.tanh(self.X)) ** 2
        return grad_output * grad_tanh
    
if __name__ == "__main__":
    tanh = Tanh()
    x = np.random.randn(2, 10)
    print("x: ", x)
    out = tanh(x)
    print("out: ", out)
    d_out = np.random.randn(2, 10)
    grads = tanh.backward(d_out)
    print("grad: ", grads)