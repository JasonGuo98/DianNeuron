from .activations_class import *

activations_list = [None, "softmax", "relu", "sigmoid", "tanh", "ELU", "swish", "LeakyReLU"]


def get_activation(name, forward=None, backward=None):
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
        elif name == "ELU":
            return ELU()
        elif name == "swish":
            return Swish()
        elif name == "LeakyReLU":
            return LeakyReLU()
        else:
            return Activation()
    elif forward and backward:
        # 这个是返回一个激活函数类的方法，需要手动构造正向和反向过程
        activation = Activation(name)
        activation.forward = forward
        activation.backward = backward
        return activation
    else:
        try:
            if name.TYPE == "activation":
                return name  # 说明传入的name已经是一个激活函数了
            else:
                raise ValueError("Incorrect activation, try:", activations_list)
        except:
            raise ValueError("Incorrect activation, try:", activations_list)
