import numpy as np

class Flatten():
    def __init__(self):
        pass

    def __call__(self, x):
        self.x_shape = x.shape
        return self.forward(x)
    
    def forward(self, x):
        out = x.ravel().reshape(self.x_shape[0], -1)
        return out
    
    def backward(self, d_out):
        d_x = d_out.reshape(self.x_shape)
        return d_x
    
if __name__ == "__main__":
    flatten = Flatten()
    x = np.random.randn(2, 10)
    print("x: ", x)
    out = flatten(x)
    print("out: ", out)
    d_out = np.random.randn(2, 10)
    grads = flatten.backward(d_out)
    print("grad: ", grads)