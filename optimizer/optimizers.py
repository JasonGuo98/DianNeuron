import numpy as np
from .optimizers_class import *
optimizers_list = ['sgd','adam']

def get_optimizer(optimizer_name,learning_rate = 1e-3):
    if  optimizer_name in optimizers_list :
        if optimizer_name == 'sgd':
            return Sgd(learning_rate = learning_rate)
        elif optimizer_name == "adam":
            return Adam(learning_rate = learning_rate)
    else:
        try:
            if optimizer_name.TYPE == "optimizer":
                return optimizer_name
            else:
                raise ValueError("optimizer must in :",optimizers_list )
        except:
            raise ValueError("optimizer must in :",optimizers_list )