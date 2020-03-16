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

        # 这里需要一个图裁剪算法，自动将输入和输出的子图裁剪出来
        # 这里需要一个获取有向无环图的出入度的算法，使得build和结构定义可以分开

        self.all_pamaters = []
        self.forward_priority,self.forward = get_forward_func(input_layer,output_layer)
        # 获取前向传播函数
        self.backward_priority,self.backward = get_backword_func(input_layer,output_layer)
        # 获取反向传播函数

        for layer in self.forward_priority:
            if layer.TYPE == 'layer':
                # 只有layer有参数
                for parameter in layer.parameters.values():
                    self.all_pamaters.append(parameter)
        

        # 将所有相关变量放入优化器中
        self.optimizer.add_parameters(self.all_pamaters)
        # 将变量加入loss中，为了计算他们的正则化，这里的正则化已经加入到参数内部了
        self.lossfunction.add_parameters(self.all_pamaters)
        # print("before get forward")
        

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
    # 目前只能有一个输入，和一个输出

    def get_priority(input_layer,output_layer):
        zero_in_degree_nodes = [input_layer]
        priority = []
        for layer in zero_in_degree_nodes:
            priority.append(layer)
            for next_layer in layer.next_layer_list:
                next_layer.in_degree-=1
                if next_layer.in_degree == 0 and next_layer not in zero_in_degree_nodes:
                    zero_in_degree_nodes.append(next_layer)
        return priority
    priority = get_priority(input_layer,output_layer)
    def forward_func(x):
        # 现在可以处理有向无环图图
        priority[0].forward(x)
        for layer in priority[1:]:
            layer.auto_forward()
        y=output_layer.info_dic['y']
        return y
    # def forward_func(x):
    #     if input_layer != output_layer:
    #         # 递归查找
    #         return output_layer.forward(get_forward_func(input_layer,output_layer.last_layer)(x))
    #     else:
    #         return input_layer.forward(x)
    return priority,forward_func

def get_backword_func(input_layer,output_layer):
    # 输出出度不一定为0
    # 这里要求至少有一个为0
    def get_priority(input_layer,output_layer):
        zero_out_degree_nodes = [output_layer]
        priority = []
        
        for layer in zero_out_degree_nodes:
            priority.append(layer)
            if layer.TYPE == 'op':
                # op可以有多个入度
                for last_layer in layer.last_layers_list:
                    last_layer.out_degree-=1
                    if last_layer.out_degree == 0 and last_layer not in zero_out_degree_nodes:
                        zero_out_degree_nodes.append(last_layer)
            elif layer.TYPE == 'layer':
                # layer 只有一入出度
                if layer.last_layer:
                    # 不为输入
                    last_layer = layer.last_layer
                    last_layer.out_degree-=1
                    if last_layer.out_degree == 0 and last_layer not in zero_out_degree_nodes:
                        zero_out_degree_nodes.append(last_layer)
        return priority
    priority = get_priority(input_layer,output_layer)
    def backward_func(grid_on_y):
        # 现在可以处理有向无环图图
        priority[0].backward(grid_on_y, )
        for layer in priority[1:]:
            layer.auto_backward()
        return input_layer.grid_on_x


    # def backward_func(grid_on_y):
    #     if input_layer != output_layer:
    #         # 递推回溯
    #         return get_backword_func(input_layer,output_layer.last_layer)(output_layer.backward(grid_on_y))
    #     else:
    #         return input_layer.backward(grid_on_y)
    return priority,backward_func
