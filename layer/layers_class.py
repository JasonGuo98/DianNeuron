import numpy as np
from ..parameter.parameter import Parameter
from ..activation.activations import activations_list,get_activation

class Layer(object):
    TYPE = "layer"
    def __init__(self,):
        pass
    def forward(self,x):
        pass
    def backward(self,grid_on_y):
        pass
    def auto_forward(self,):
        pass
    def auto_backward(self,):
        pass

class Inputs(Layer):
    count = 0
    def __init__(self,out_dim,name):
        super(Inputs, self).__init__()
        self.out_dim = self.in_dim = out_dim
        # 输入layer的入度为0，出度为N
        self.next_layer_list = []
        self.in_degree = 0
        self.out_degree = 0
        self.last_layer = None
        self.parameters = {}

        if name:
            self.name = name+str(out_dim)+" : %d"%Inputs.count
        else:
            self.name = "Inputs"+str(out_dim)+" : %d"%Inputs.count
        self.grid_on_x = np.zeros(out_dim)
        self.info_dic = {}

        Inputs.count+=1


    def forward(self,x):
        if x.shape[1:] != (self.in_dim,):
            raise ValueError("inputs dim: ",x.shape[1:]," and the build in_dim: ",self.out_dim,"does not match!")
        self.info_dic['y'] = x

        return x
    def auto_forward(self,):
        # 臣妾做不到
        return 
        
    def backward(self,grid_on_y):
        self.grid_on_x = grid_on_y
        # 记录对x的梯度，但不输出
        # return grid_on_x
        return grid_on_x

    def auto_backward(self,):
        grid_list_on_y = []
        grid_on_y = self.next_layer_list[0].info_dic['grid_on_%s'%self.name]
        for layer in self.next_layer_list[1:]:
            grid_on_y+=layer.info_dic['grid_on_%s'%self.name]
        self.grid_on_x = grid_on_y
        return self.grid_on_x

class Dense(Layer):
    count = 0
    def __init__(self,last_layer,out_dim,activation,W_regularization,W_regularizationRate,W_init,b_init,name):
        super(Dense, self).__init__()
        last_layer.next_layer_list.append(self)
        last_layer.out_degree+=1
        self.next_layer_list = []
        self.in_degree = 1
        self.out_degree = 0
        # 认为，一个layer的入度，只能是1，而出度可以是N

        if activation == 'softmax' and out_dim==1:
            raise ValueError("for softmax activation, out_dim can't be 1!")
        self.activation = get_activation(activation)


        self.last_layer = last_layer
        self.in_dim = last_layer.out_dim
        self.out_dim = out_dim
        if name:
            self.name = name+str(out_dim)+" : %d"%Dense.count
        else:
            self.name = "Dense"+str(out_dim)+" : %d"%Dense.count
        pass

        self.W = Parameter((self.in_dim,self.out_dim),name = self.name+":W", \
            regularization = W_regularization,regularizationRate = W_regularizationRate,init =W_init) 
        self.b = Parameter((self.out_dim,),name = self.name+"：b",init=b_init)
        self.parameters = {"W":self.W,'b':self.b}
        
        self.info_dic = {}
        Dense.count+=1

        

    def forward(self,x):
        if x.shape[1:] != (self.in_dim,):
            raise ValueError("inputs dim: ",x.shape[1:]," and the build in_dim: ",self.in_dim,"does not match!")
        wxb = x @ self.W.value+self.b.value

        y = self.activation.forward(wxb)

        # 保留必要的中间值方便计算梯度
        self.info_dic['y'] = y
        self.info_dic['wxb'] = wxb
        self.info_dic['x'] = x
        
        # for parameter in self.parameters.value():
        #     parameter.cal_regularization()
        self.W.cal_regularization()
        self.b.cal_regularization()
        return y
    def auto_forward(self,):
        self.forward(self.last_layer.info_dic['y'])

        
    def backward(self,grid_on_y):
        # 只用处理单输出
        grid_on_xwb = self.activation.backward(grid_on_y,self.info_dic)

        x = self.info_dic['x']
        grid_on_x = grid_on_xwb@self.W.value.T
        grid_on_W = x.T@grid_on_xwb
        grid_on_b = np.sum(grid_on_xwb,axis=0)

        self.W.gradient = grid_on_W
        self.b.gradient = grid_on_b
        self.info_dic['grid_on_%s'%self.last_layer.name] = grid_on_x

        return grid_on_x
    def auto_backward(self,):
        grid_on_y = self.next_layer_list[0].info_dic['grid_on_%s'%self.name]
        for layer in self.next_layer_list[1:]:
            grid_on_y+=layer.info_dic['grid_on_%s'%self.name]
        grid_on_x = self.backward(grid_on_y)
