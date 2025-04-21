import numpy as np

class Sigmoid():
    def __init__(self):
        self.X = None

    def __call__(self, X):
        self.X = X
        return self.forward(self.X)
    
    def forward(self, X):
        return self._sigmoid(X)
    
    def backward(self, grad_output):
        sigmoid_grad = self._sigmoid(self.X) * (1 - self._sigmoid(self.X))
        return grad_output * sigmoid_grad
    
    def _sigmoid(self, X):
        return 1.0 / (1 + np.exp(-X))
    
if __name__ == "__main__":
    sig = Sigmoid()
    x = np.random.randn(2, 10)
    print("x: ", x)
    out = sig(x)
    print("out: ", out)
    d_out = np.random.randn(2, 10)
    grads = sig.backward(d_out)
    print("grad: ", grads)