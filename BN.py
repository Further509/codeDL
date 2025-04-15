import numpy as np

class BatchNorm:
    def __init__(self, momentum=0.01, eps=1e-5, feat_dim=2, training=True):
        '''
        momentum 动量 计算每个batch均值和方差的滑动均值
        eps 防止分母为0
        feat_dim 特征维度
        '''
        self.training = training
        self._running_mean = np.zeros(shape=(feat_dim, ))
        self._running_var = np.ones(shape=(feat_dim, ))

        self._momentum = momentum
        self._eps = eps
        self._beta = np.zeros(shape=(feat_dim, ))
        self._gamma = np.zeros(shape=(feat_dim, ))
    
    def batch_norm(self, x):
        if self.training:
            x_mean = x.mean(axis=0)
            x_var = x.var(axis=0)
            self._running_mean = (1 - self._momentum) * x_mean + self._momentum * self._running_mean
            self._running_var = (1 - self._momentum) * x_var + self._momentum * self._running_var
            x_hat = (x - x_mean) / np.sqrt(x_var + self._eps)
        else:
            x_hat = (x - self._running_mean) / np.sqrt(self._running_var + self._eps)

        return self._gamma * x_hat + self._beta
    
if __name__ == "__main__":
    np.random.seed(0)
    data = np.random.randn(3, 2)
    BN = BatchNorm(training=True)
    output = BN.batch_norm(data)
    print(output)
