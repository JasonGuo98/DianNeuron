import numpy as np
from .activations_class import *
activations_list = [None,"softmax","relu","sigmoid","tanh"]

def get_activation(name,forward=None,backword=None):
    if name in activations_list:
        if name == "softmax":
            # print("get_softmax")
            return Softmax()
        elif name == "relu":
            return Relu()
        elif name == "sigmoid":
            return Sigmoid()
        elif name == "tanh":
            return Tanh()
        else:
            return Activation()
    elif forward and backword:
        # 这个是返回一个激活函数类的方法，需要手动构造正向和反向过程
        activation = Activation(name)
        activation.forward = forward
        activation.backword = backword
        return activation
    else:
        try :
            if name.TYPE == "activation":
                return name #说明传入的name已经是一个激活函数了
            else:
                raise ValueError("Incorrect activation, try:",activations_list)
        except:
            raise ValueError("Incorrect activation, try:",activations_list)