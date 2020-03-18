from .. import *


class Optimizer(object):
    """docstring for Loss"""
    TYPE = 'optimizer'

    def __init__(self, name, learning_rate):
        self.name = name
        self.learning_rate = learning_rate

    def add_parameters(self, all_parameters):
        self.all_parameters = all_parameters

    def optimize(self):
        # print("do optimize")
        pass


class Sgd(Optimizer):
    def __init__(self, name="sgd", learning_rate=1e-3):
        self.name = name
        self.learning_rate = learning_rate

    def optimize(self, t):
        # print("do optimize sgd")
        for parameter in self.all_parameters:
            parameter.value -= (parameter.gradient + parameter.reg_gradient) * self.learning_rate
            # parameter.value_check()


class Adam(Optimizer):
    def __init__(self, name='adam', learning_rate=1e-3, alpha=0.9, beta=0.999, delta=1e-8, ):
        self.name = name
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.s = {}  # 一阶矩
        self.r = {}  # 二阶矩

    def add_parameters(self, all_parameters):
        learning_rate = self.learning_rate
        alpha = self.alpha
        beta = self.beta
        delta = self.delta

        self.all_parameters = all_parameters
        for parameter in all_parameters:
            # print(type(parameter.value))
            # print(parameter.value.shape)
            # s = 
            # 这里用zeroslike会在cupy报错
            self.s[parameter.name] = np.zeros_like(parameter.value)  # 一阶矩
            # r = 
            self.r[parameter.name] = np.zeros_like(parameter.value)  # 二阶矩

    def optimize(self, t):
        learning_rate = self.learning_rate
        alpha = self.alpha
        beta = self.beta
        delta = self.delta
        for parameter in self.all_parameters:
            self.s[parameter.name] *= alpha
            self.s[parameter.name] += (1 - alpha) * (parameter.gradient + parameter.reg_gradient)
            # self.s[parameter.name] = alpha*self.s[parameter.name] + (1-alpha)*(parameter.gradient+parameter.reg_gradient)
            self.r[parameter.name] *= beta
            self.r[parameter.name] += (1 - beta) * ((parameter.gradient + parameter.reg_gradient) ** 2)
            # self.r[parameter.name] = beta*self.r[parameter.name] + (1-beta)*((parameter.gradient+parameter.reg_gradient)**2)
            s_ = self.s[parameter.name] / (1 - alpha ** t)
            r_ = self.r[parameter.name] / (1 - beta ** t)
            gradient = s_ / (np.sqrt(r_) + delta)
            parameter.value -= gradient * learning_rate
            del gradient, s_, r_


class Momentum(Optimizer):
    def __init__(self, name='momentum', learning_rate=1e-3, momentum=0.9):
        self.name = name
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = {}

    def optimize(self, t):
        # v = - dx * lr + v * momentum,先更新v再梯度变化
        for parameter in self.all_parameters:
            self.v = -(parameter.gradient + parameter.reg_gradient) * self.learning_rate + self.v * self.momentum
            parameter.value += self.v

# class Adam(Optimizer):
#     def __init__(self, name='Adam', learning_rate=1e-3, beta1=1e-8, beta2=0.999, esp=1e-8):
#         self.name = name
#         self.learning_rate = learning_rate
#         self.esp = esp
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.m = {}
#         self.v = {}
#
#     def optimize(self, t):
#         # m = beta1*m + (1-beta1)*dx
#         # v = beta2*v + (1-beta2)*(dx**2)
#         # x += - learning_rate * m / (np.sqrt(v) + eps)
#         for parameter in self.all_parameters:
#             dx = parameter.gradient + parameter.reg_gradient
#             self.m = self.beta1 * self.m + (1-self.beta1) * dx
#             self.v = self.beta2 * self.v + (1-self.beta2) * (dx**2)
#             parameter.value += -self.learning_rate * self.m / (np.sqrt(self.v) + self.esp)
