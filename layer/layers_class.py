import numpy as np
from ..parameter.parameter import Parameter
from ..activation.activations import activations_list,get_activation

class Layer(object):
    TYPE = "layer"
    def __init__(self,):
        pass
    def forward(self,):
        pass
    def backward(self,):
        pass

class Inputs(Layer):
    count = 0
    def __init__(self,out_dim,name):
        super(Inputs, self).__init__()
        self.out_dim = self.in_dim = out_dim
        if name:
            self.name = name+str(out_dim)+" : %d"%Inputs.count
        else:
            self.name = "Inputs"+str(out_dim)+" : %d"%Inputs.count
        self.grid_on_x = np.zeros(out_dim)
        Inputs.count+=1

    def forward(self,x):
        if x.shape[1:] != (self.in_dim,):
            raise ValueError("inputs dim: ",x.shape[1:]," and the build in_dim: ",self.out_dim,"does not match!")
        return x
        
    def backward(self,grid_on_y):
        self.grid_on_x = grid_on_y
        # 记录对x的梯度，但不输出
        # return grid_on_x
        return grid_on_y

class Dense(Layer):
    count = 0
    def __init__(self,last_layer,out_dim,activation,W_regularization,W_regularizationRate,W_init,b_init,name):
        super(Dense, self).__init__()

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
        delta = 1e-3
        if x.shape[1:] != (self.in_dim,):
            raise ValueError("inputs dim: ",x.shape[1:]," and the build in_dim: ",self.in_dim,"does not match!")
        wxb = x @ self.W.value+self.b.value

        y = self.activation.forward(wxb)


        # if self.activation == "softmax":
        #     wxb-=np.max(wxb,axis = 1,keepdims=True)
        #     exp = np.exp(wxb)+delta
        #     exp_sum = np.sum(exp,axis=1,keepdims=True)+delta*self.out_dim
        #     # 这样使得最小值不会小于delta，最大值不会大于1-delta*(self.out_dim-1)
        #     # 当输出维度为1的时候，输出1，不允许这样做
        #     y = exp/exp_sum

        # elif self.activation == "relu":
        #     y = wxb
        #     y[y<0] = 0
        # elif self.activation == "tanh":# = 2sigmoid(x)-1
        #     y = 2/(1+np.exp(-2*wxb)) - 1 
        # elif self.activation == 'sigmoid':
        #     y = 1/(1+np.exp(-wxb))
        # else:
        #     y = wxb

        # 保留必要的中间值方便计算梯度
        self.info_dic['y'] = y
        self.info_dic['wxb'] = wxb
        self.info_dic['x'] = x
        
        # for parameter in self.parameters.value():
        #     parameter.cal_regularization()
        self.W.cal_regularization()
        self.b.cal_regularization()
        return y
        
    def backward(self,grid_on_y):

        grid_on_xwb = self.activation.backward(grid_on_y,self.info_dic)
        # if self.activation == "softmax":
        #     y = self.info_dic['y']
        #     out_dim = self.out_dim
        #     ones = (np.ones((len(y),1,1))*np.eye(out_dim).reshape(1,out_dim,out_dim))
        #     dy_dwxb = (y.reshape(-1,out_dim,1)*(ones-y.reshape(-1,1,out_dim)))
        #     grid_on_xwb = np.sum(grid_on_y.reshape(-1,1,out_dim)@dy_dwxb,axis=-2).reshape(-1,out_dim)
        # elif self.activation == "relu":
        #     grid_on_xwb = grid_on_y
        #     y = self.info_dic['y']
        #     grid_on_xwb[y == 0] = 0
        # elif self.activation == "tanh":
        #     y = self.info_dic['y']
        #     grid_on_xwb = grid_on_y*(1-y**2)
        # elif self.activation == 'sigmoid':
        #     y = self.info_dic['y']
        #     grid_on_xwb = grid_on_y*y*(1-y)
        # else:
        #     grid_on_xwb = grid_on_y

        x = self.info_dic['x']
        grid_on_x = grid_on_xwb@self.W.value.T
        grid_on_W = x.T@grid_on_xwb
        grid_on_b = np.sum(grid_on_xwb,axis=0)

        self.W.gradient = grid_on_W
        self.b.gradient = grid_on_b

        return grid_on_x
