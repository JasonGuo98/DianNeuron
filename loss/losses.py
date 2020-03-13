import numpy as np
import copy
from .losses_class import *

losses_list = ["CrossEntropy","HingeLoss"]


def get_lossfunction(lossfunction_name,cal_loss_and_grid = None):
    if lossfunction_name in losses_list:
        if lossfunction_name == "CrossEntropy":
            return CrossEntropy()
        elif lossfunction_name == "HingeLoss":
            return HingeLoss()
    elif cal_loss_and_grid:
        # 加入可以拓展的lossfunction接口
        lossfunction = Loss(name)
        lossfunction.cal_loss_and_grid = cal_loss_and_grid
        return lossfunction
    else:
        try :
            if name.TYPE == "lossfunction":
                return name # 说明name 已经是一个lossfunction
            else:
                raise ValueError("classification loss must in :",losses_list)
        except:
            raise ValueError("classification loss must in :",losses_list)
    