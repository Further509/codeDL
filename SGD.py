import numpy as np

class SGD():
    def __init__(self, parameters, lr, momentum=None):
        '''
        parameters: 模型训练的参数
        lr: 学习率
        momentum: 动量因子
        '''
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum

        if momentum is not None:
            self.velocity = self.velocity_initial()

    def update_parameters(self, grads):
        if self.momentum is None:
            for param, grad in zip(self.parameters, grads):
                param -= self.lr * grad
        else:
            for i in range(len(self.parameters)):
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
                self.parameters[i] += self.velocity[i]

    def velocity_initial(self):
        '''
        将velocity参数初始化为0
        '''
        velocity = []
        for param in self.parameters:
            velocity.append(np.zeros_like(param))
        return velocity
    
if __name__ == "__main__":
    sgd = SGD([np.array([1.0, 2.0]), np.array([3.0, 4.0])], lr=0.01, momentum=0.9)
    print(sgd.parameters)
    print(sgd.velocity)
    sgd.update_parameters([np.array([0.1, 0.2]), np.array([0.3, 0.4])])
    print(sgd.parameters)
    print(sgd.velocity)