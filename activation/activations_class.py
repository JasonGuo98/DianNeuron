# coding=utf-8
from .. import *
import copy


class Activation(object):
    """docstring for Activation"""
    TYPE = "activation"

    def __init__(self, name="liner"):
        self.name = name

    def forward(self, x):
        return x

    def backward(self, grid_on_y, info_dic=None):
        return grid_on_y


class Relu(Activation):
    """docstring for Activation"""

    def __init__(self, name='relu'):
        self.name = name

    def forward(self, x):
        y = copy.deepcopy(x)
        y[y < 0] = 0
        return y

    def backward(self, grid_on_y, info_dic=None):
        y = info_dic['y']
        grid_on_x = copy.deepcopy(grid_on_y)
        grid_on_x[y == 0] = 0

        return grid_on_x


class Tanh(Activation):
    """docstring for Activation"""

    def __init__(self, name='tanh'):
        self.name = name

    def forward(self, x):
        y = 2 / (1 + np.exp(-2 * x)) - 1
        return y

    def backward(self, grid_on_y, info_dic=None):
        y = info_dic['y']

        return grid_on_y * (1 - y ** 2)


class Softmax(Activation):
    """docstring for Activation"""

    def __init__(self, name='softmax', delta=1e-7):
        self.name = name
        self.delta = delta

    def forward(self, x):
        batch_max = np.max(x, axis=1, keepdims=True)
        x -= batch_max
        exp = np.exp(x) + self.delta
        exp_sum = np.sum(exp, axis=1, keepdims=True)
        # exp_sum = np.sum(exp, axis=1, keepdims=True) + self.delta * x.shape[-1]
        # 这样使得最小值不会小于delta，最大值不会大于1-delta*(self.out_dim-1)
        # 当输出维度为1的时候，输出1，不允许这样做
        x += batch_max
        y = exp / exp_sum
        return y

    def backward(self, grid_on_y, info_dic=None):
        y = info_dic['y']
        out_dim = grid_on_y.shape[-1]
        ones = (np.ones((len(y), 1, 1)) * np.eye(out_dim).reshape(1, out_dim, out_dim))
        dy_dx = (y.reshape(-1, out_dim, 1) * (ones - y.reshape(-1, 1, out_dim)))
        grid_on_x = np.sum(grid_on_y.reshape(-1, 1, out_dim) @ dy_dx, axis=-2).reshape(-1, out_dim)
        return grid_on_x


class Sigmoid(Activation):
    """docstring for Activation"""

    def __init__(self, name='sigmoid'):
        self.name = name

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, grid_on_y, info_dic=None):
        y = info_dic['y']
        grid_on_x = grid_on_y * y * (1 - y)
        return grid_on_x


class ELU(Activation):

    def __init__(self, name='ELU', alpha=0.01):
        self.name = name
        self.alpha = alpha

    def forward(self, x):
        y = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        return y

    def backward(self, grid_on_y, info_dic=None):
        x = info_dic['wxb']
        return grid_on_y * np.where(x > 0, 1, self.alpha * np.exp(x))


class Swish(Activation):

    def __init__(self, name="swish", beta=1):
        self.name = name
        self.beta = beta

    def forward(self, x):
        y = x * (np.exp(self.beta * x) / (np.exp(self.beta * x) + 1))
        return y

    def backward(self, grid_on_y, info_dic=None):
        y = info_dic['y']
        return np.exp(self.beta * x) / (1 + np.exp(self.beta * x)) + x * (
                self.beta * np.exp(self.beta * x) / ((1 + np.exp(self.beta * x)) * (1 + np.exp(self.beta * x))))
