try:
    import cupy as np
except:
    import numpy as np

import copy

class Activation(object):
    """docstring for Activation"""
    TYPE = "activation"
    def __init__(self, name = "liner"):
        self.name = name

    def forward(self,x):
        return x

    def backward(self,grid_on_y,info_dic = None):
        return grid_on_y

class Relu(object):
    """docstring for Activation"""
    def __init__(self, name='relu'):
        self.name = name

    def forward(self,x):
        y = copy.deepcopy(x)
        y[y<0] = 0
        return y

    def backward(self,grid_on_y,info_dic):
        y = info_dic['y']
        grid_on_x = copy.deepcopy(grid_on_y)
        grid_on_x[y == 0] = 0

        return grid_on_x

class Tanh(object):
    """docstring for Activation"""
    def __init__(self, name='tanh'):
        self.name = name

    def forward(self,x):
        y = 2/(1+np.exp(-2*x)) - 1
        return y

    def backward(self,grid_on_y,info_dic):
        y = info_dic['y']

        return grid_on_y*(1-y**2)

class Softmax(object):
    """docstring for Activation"""
    def __init__(self, name='softmax',delta = 1e-6):
        self.name = name
        self.delta = delta

    def forward(self,x):
        x-=np.max(x,axis = 1,keepdims=True)
        exp = np.exp(x)+self.delta
        exp_sum = np.sum(exp,axis=1,keepdims=True)+self.delta*x.shape[-1]
        # 这样使得最小值不会小于delta，最大值不会大于1-delta*(self.out_dim-1)
        # 当输出维度为1的时候，输出1，不允许这样做
        y = exp/exp_sum
        return y

    def backward(self,grid_on_y,info_dic):
        y = info_dic['y']
        out_dim = grid_on_y.shape[-1]
        ones = (np.ones((len(y),1,1))*np.eye(out_dim).reshape(1,out_dim,out_dim))
        dy_dx = (y.reshape(-1,out_dim,1)*(ones-y.reshape(-1,1,out_dim)))
        grid_on_x = np.sum(grid_on_y.reshape(-1,1,out_dim)@dy_dx,axis=-2).reshape(-1,out_dim)
        return grid_on_x

class Sigmoid(object):
    """docstring for Activation"""
    def __init__(self, name = 'sigmoid'):
        self.name = name

    def forward(self,x):
        y = 1/(1+np.exp(-x))
        return y

    def backward(self,grid_on_y,info_dic):
        y = info_dic['y']
        grid_on_x = grid_on_y*y*(1-y)
        return grid_on_x


def get_activation(name,forward=None,backword=None):
    activations_list = ["softmax","relu","sigmoid","tanh"]
    if name in activations_list:
        if name == "softmax":
            return Softmax()
        elif name == "relu":
            return Relu()
        elif name == "sigmoid":
            return Sigmoid()
        elif name == "tanh":
            return Tanh()
    elif forward and backword:
        activation = Activation()
        activation.forward = forward
        activation.backword = backword
        return activation
    else:
        raise ValueError("Incorrect activation")

