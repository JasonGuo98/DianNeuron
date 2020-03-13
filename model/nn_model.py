import numpy as np
from functools import wraps
import copy
from utils.mini_batch_iter import MINI_BATCH_ITER
from .classifier import Classifier
from ..parameter.parameter import Parameter
from ..layer.layers import *
from ..loss.losses import *
from ..optimizer.optimizers import *

class NN_Model(object):
    """docstring for NN_MODEL"""
    TYPE = "MODEL"
    def __init__(self, input_layer,output_layer,optimizer = "sgd",learning_rate = 1e-3,\
                                    lossfunction = "CrossEntropy"):
        print("build model")
        
        if type(lossfunction) ==str :
            if lossfunction not in ["CrossEntropy","HingeLoss"]:
                raise ValueError("classification loss must in :",["CrossEntropy","HingeLoss"])
            else:
                self.lossfunction = get_lossfunction(lossfunction)
        elif lossfunction.TYPE == "lossfunction":
            self.lossfunction = lossfunction
        else:
            raise ValueError("Wrong lossfunction")

        if type(optimizer)==str:
            if  optimizer not in ['sgd','adam']:
                raise ValueError("optimizer must in :",['sgd','adam'])
            else:
                self.optimizer = get_optimizer(optimizer,learning_rate)
        elif optimizer.TYPE == 'optimizer':
            self.optimizer = optimizer
        else:
            raise ValueError("Wrong optimizer")

        self.all_pamaters = []
        layer_now = output_layer

        # 目前遍历方法只能支持无分支模型
        while(layer_now!=input_layer):
            for parameter in layer_now.parameters.values():
                self.all_pamaters.append(parameter)
            layer_now = layer_now.last_layer

        # 将所有相关变量放入优化器中
        self.optimizer.add_parameters(self.all_pamaters)
        # 将变量加入loss中，为了计算他们的正则化，这里的正则化已经加入到参数内部了
        self.lossfunction.add_parameters(self.all_pamaters)
        # print("before get forward")
        self.forward = get_forward_func(input_layer,output_layer)
        # 获取前向传播函数
        self.backward = get_backword_func(input_layer,output_layer)
        # 获取反向传播函数

        self.step = 0 # 起始时间步为0


    def cal_loss_and_grid(self,y_pred,y_batch):
        # 返回一个batch上的平均loss
        # 以及在输出上的梯度
        return self.lossfunction.cal_loss_and_grid(y_pred,y_batch)

    def train_on_batch(self,x_batch,y_batch):
        self.step+=1
        y_pred = self.forward(x_batch)
        loss_on_batch,reg_loss,grid_on_y_pred = self.cal_loss_and_grid(y_pred,y_batch)
        # print(grid_on_y_pred)
        grid_on_input = self.backward(grid_on_y_pred)
        self.optimizer.optimize(t = self.step)
        return loss_on_batch,reg_loss,grid_on_input


# 这里需要一种新的递归遍历方式
def get_forward_func(input_layer,output_layer):
    # 这里的语法有问题
    # print("do get forward")
    # if type(input_layers) != list:
    #     input_layers = [input_layers]
    # if type(output_layers) != list:
    #     output_layers = output_layers

    # def cal_priority(input_layers,output_layers):
    #     zero_in_degree_nodes = input_layers
    #     priority = []
    #     for layer in zero_in_degree_nodes:
    #         priority.append(layer)
    #         for next_layer in layer.next_layers:
    #             next_layer.in_degree-=1
    #             if next_layers.in_degree == 0:
    #                 zero_in_degree_nodes.append(next_layer)
    #     return priority
    # priority = get_priority(input_layers,output_layers)
    # def forward_func(x_list):
    #     # 现在可以处理有向无环图图
    #     for i in range(len(x_list)):
    #         priority[i].forward(x[i])
    #     for layer in priority[len(x_list):]:
    #         layer.auto_forward()
    #     y_list = []
    #     for layer in output_layer:
    #         y_list.append(layer.info_dic['y'])
    #     return y_list
    def forward_func(x):
        if input_layer != output_layer:
            # 递归查找
            return output_layer.forward(get_forward_func(input_layer,output_layer.last_layer)(x))
        else:
            return input_layer.forward(x)
    return forward_func

def get_backword_func(input_layer,output_layer):
    # 输出出度不一定为0
    # 这里要求至少有一个为0
    # def cal_priority(input_layers,output_layers):
    #     zero_out_degree_nodes = []
    #     priority = []
    #     for layer in output_layers:
    #         if layer.out_degree == 0:
    #             zero_out_degree_nodes.append(layer)
    #     for layer in zero_out_degree_nodes:
    #         priority.append(layer)
    #         for last_layer in layer.last_layers:
    #             last_layer.out_degree-=1
    #             if last_layers.out_degree == 0:
    #                 zero_out_degree_nodes.append(last_layer)
    #     return priority
    # priority = get_priority(input_layers,output_layers)
    # def backward_func(grid_info_dic_list):
    #     # 现在可以处理有向无环图图
    #     for i in range(len(grid_info_dic_list)):
    #         priority[i].forward(x[i])
    #     for layer in priority[len(x_list):]:
    #         layer.auto_forward()
    #     y_list = []
    #     for layer in output_layer:
    #         y_list.append(layer.info_dic['y'])
    #     return y_list


    def backward_func(grid_on_y):
        if input_layer != output_layer:
            # 递推回溯
            return get_backword_func(input_layer,output_layer.last_layer)(output_layer.backward(grid_on_y))
        else:
            return input_layer.backward(grid_on_y)
    return backward_func
