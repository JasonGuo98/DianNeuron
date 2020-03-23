from .. import *
from ..parameter.parameter import Parameter
from ..activation.activations import activations_list, get_activation


class Layer(object):
    TYPE = "layer"
    name = "layer"
    def __init__(self,):
        pass
    def forward(self,x,is_train = True):
        pass

    def backward(self, grid_on_y):
        pass

    def auto_forward(self,is_train = True):
        pass

    def auto_backward(self, ):
        pass


class Inputs(Layer):
    count = 0
    name = "Inputs"
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
            self.name = name + str(out_dim) + " : %d" % Inputs.count
        else:
            self.name = self.name+str(out_dim)+" : %d"%Inputs.count

        self.grid_on_x = np.zeros(out_dim)
        self.info_dic = {}

        Inputs.count += 1

    def forward(self,x,is_train = True):
        if x.shape[1:] != (self.in_dim,):
            raise ValueError("inputs dim: ", x.shape[1:], " and the build in_dim: ", self.out_dim, "does not match!")
        self.info_dic['y'] = x

        return x
    def auto_forward(self,is_train = True):
        # 臣妾做不到
        return

    def backward(self, grid_on_y):
        self.grid_on_x = grid_on_y
        # 记录对x的梯度，但不输出
        # return grid_on_x
        return self.grid_on_x

    def auto_backward(self, ):
        grid_list_on_y = []
        grid_on_y = self.next_layer_list[0].info_dic['grid_on_%s' % self.name]
        for layer in self.next_layer_list[1:]:
            grid_on_y += layer.info_dic['grid_on_%s' % self.name]
        self.grid_on_x = grid_on_y
        return self.grid_on_x


class Dense(Layer):
    count = 0
    name = "Dense"
    def __init__(self,last_layer,out_dim,activation,W_regularization,W_regularizationRate,W_init,b_init,dtype,name):
        super(Dense, self).__init__()
        last_layer.next_layer_list.append(self)
        last_layer.out_degree += 1
        self.next_layer_list = []
        self.in_degree = 1
        self.out_degree = 0
        # 认为，一个layer的入度，只能是1，而出度可以是N

        if activation == 'softmax' and out_dim == 1:
            raise ValueError("for softmax activation, out_dim can't be 1!")
        self.activation = get_activation(activation)

        self.last_layer = last_layer
        self.in_dim = last_layer.out_dim
        self.out_dim = out_dim
        if name:
            self.name = name + str(out_dim) + " : %d" % Dense.count
        else:
            self.name = self.name+str(out_dim)+" : %d"%Dense.count
        pass

        self.W = Parameter((self.in_dim, self.out_dim), name=self.name + ":W", \
                           regularization=W_regularization, regularizationRate=W_regularizationRate, init=W_init,
                           dtype=dtype)
        self.b = Parameter((self.out_dim,), name=self.name + "：b", init=b_init, dtype=dtype)
        self.parameters = {"W": self.W, 'b': self.b}

        self.info_dic = {}
        Dense.count += 1

    def forward(self,x,is_train = True):
        if x.shape[1:] != (self.in_dim,):
            raise ValueError("inputs dim: ", x.shape[1:], " and the build in_dim: ", self.in_dim, "does not match!")
        wxb = x @ self.W.value + self.b.value

        y = self.activation.forward(wxb)

        # 保留必要的中间值方便计算梯度
        try:
            del self.info_dic['y']
        except:
            pass
        self.info_dic['y'] = y
        try:
            del self.info_dic['wxb']
        except:
            pass
        self.info_dic['wxb'] = wxb
        try:
            del self.info_dic['x']
        except:
            pass
        self.info_dic['x'] = x

        # for parameter in self.parameters.value():
        #     parameter.cal_regularization()
        self.W.cal_regularization()
        self.b.cal_regularization()
        return y

    def auto_forward(self,is_train = True):
        self.forward(self.last_layer.info_dic['y'])

    def backward(self, grid_on_y):
        # 只用处理单输出
        grid_on_xwb = self.activation.backward(grid_on_y, self.info_dic)

        x = self.info_dic['x']
        grid_on_x = grid_on_xwb @ self.W.value.T
        grid_on_W = x.T @ grid_on_xwb
        grid_on_b = np.sum(grid_on_xwb, axis=0)

        self.W.gradient = grid_on_W
        self.b.gradient = grid_on_b
        self.info_dic['grid_on_%s' % self.last_layer.name] = grid_on_x

        return grid_on_x

    def auto_backward(self, ):
        grid_on_y = self.next_layer_list[0].info_dic['grid_on_%s' % self.name]
        for layer in self.next_layer_list[1:]:
            grid_on_y += layer.info_dic['grid_on_%s' % self.name]
        grid_on_x = self.backward(grid_on_y)


class Dropout(Layer):
    count = 0
    name = "Dropout"
    def __init__(self,last_layer,keep_prob,scale_train,name):
        super(Dropout, self).__init__()
        last_layer.next_layer_list.append(self)
        last_layer.out_degree+=1
        self.next_layer_list = []
        self.in_degree = 1
        self.out_degree = 0
        # 认为，一个layer的入度，只能是1，而出度可以是N

        self.last_layer = last_layer
        self.in_dim = last_layer.out_dim
        self.out_dim = self.in_dim
        self.keep_prob = keep_prob
        self.scale_train = scale_train

        if name:
            self.name = name+str(self.out_dim)+" : %d"%Dropout.count
        else:
            self.name = self.name+str(self.out_dim)+" : %d"%Dropout.count
        self.parameters = {}
        self.info_dic = {}
        Dropout.count+=1

    def forward(self,x,is_train = True):
        if x.shape[1:] != (self.in_dim,):
            raise ValueError("inputs dim: ",x.shape[1:]," and the build in_dim: ",self.in_dim,"does not match!")
        if not is_train:
            if self.scale_train:
                y = x*self.keep_prob
            else:
                y = x
        else:
            drops = np.random.random(x.shape)
            drops[drops<(1-self.keep_prob)] = 0
            drops[drops>=(1-self.keep_prob)] = 1
            y = x*drops
            self.info_dic['drops'] = drops
        self.info_dic['y'] = y
        return y

    def auto_forward(self,is_train = True):
        self.forward(self.last_layer.info_dic['y'],is_train)

        
    def backward(self,grid_on_y):
        drops = self.info_dic['drops']
        grid_on_x = grid_on_y*drops

        self.info_dic['grid_on_%s'%self.last_layer.name] = grid_on_x
        return grid_on_x

    def auto_backward(self,):
        grid_on_y = self.next_layer_list[0].info_dic['grid_on_%s'%self.name]
        for layer in self.next_layer_list[1:]:
            grid_on_y+=layer.info_dic['grid_on_%s'%self.name]
        grid_on_x = self.backward(grid_on_y)

class BatchNorm1d(Layer):
    count = 0
    name = "BatchNorm1d"
    def __init__(self, last_layer, init='Xavier', momentum=0.9, eps=1e-5):
        super(self,BatchNorm1d).__init__()
        last_layer.next_layer_list.append(self)
        last_layer.out_degree+=1
        self.next_layer_list = []
        self.num_features = last_layer.out_dim
        self.out_dim = self.in_dim

        self.gamma = None
        self.beta = None
        self._cache = None
        self.params = dict()
        self.running_mean = None
        self.running_var = None
        self.momentum = momentum
        self.eps = eps
        self.init_mode = init
        if name:
            self.name = name+str(self.num_features)+" : %d"%BatchNorm1d.count
        else:
            self.name = self.name+str(self.num_features)+" : %d"%BatchNorm1d.count
        self._initialize()

    def _initialize(self):
        self.gamma = Parameter(shape=[self.num_features], name=self.name+":gamma" , init=self.init_mode)
        self.beta = Parameter(shape=[self.num_features], name=self.name+":beta", init=self.init_mode)
        self.running_mean = np.zeros([self.num_features],dtype='float32')
        self.running_var = np.zeros([self.num_features], dtype='float32')
        self.params['gamma'] = self.gamma
        self.params['beta'] = self.beta

    def forward(self, x, is_train = True):
        # x.shape: [N, D]
        if is_train:
            sample_mean = np.mean(x, axis=0)
            sample_var = np.sum(np.square((x - sample_mean)), axis=0) / x.shape[0]
            self.running_mean = sample_mean * (1 - self.momentum) + self.running_mean * self.momentum
            self.running_var = sample_var * (1 - self.momentum) + self.running_var * self.momentum
            std_var = (np.sqrt(sample_var + self.eps))
            x_ = (x - sample_mean) / std_var
            out = self.gamma.value * x_ + self.beta.value
            self._cache = x, x_, sample_mean, std_var, sample_var
        else:
            x_ = (x - self.running_mean) / (np.sqrt(self.running_var + self.eps))
            out = self.gamma.value * x_ + self.beta.value
        return out

    def backward(self, grad_in):
        x, x_, sample_mean, sqrt_var, var = self._cache
        N, D = grad_in.shape
        dx = grad_in * self.gamma.value

        dbeta = np.sum(grad_in, axis=0)
        dgamma = x_ * grad_in
        dgamma = np.sum(dgamma, 0)

        dx = (1. / N) * 1 / sqrt_var * (N * dx - np.sum(dx, axis=0) - x_ * np.sum(dx * x_, axis=0))

        self.gamma.gradient = dgamma
        self.beta.gradient = dbeta

        return dx

    def auto_forward(self,is_train = True):
        self.forward(self.last_layer.info_dic['y'],is_train)

    def auto_backward(self, ):
        pass