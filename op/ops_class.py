
class OP(object):
    """docstring for OP"""
    TYPE = "op"
    def __init__(self, arg):
        super(OP, self).__init__()
    def forward(self,):
        pass

    def backward(self,):
        pass

    def auto_forward(self,is_train):
        pass

    def auto_backward(self,is_train):
        pass
        

class ADD(OP):
    """docstring for ADD"""
    count = 0
    def __init__(self, layers_list,name):
        
        # 他的输入输出都可以是一个op，或layer，且入度出度都可以是N
        for layer in layers_list:
            layer.next_layer_list.append(self)
            layer.out_degree+=1

        self.next_layer_list = []
        self.in_degree = len(layers_list)
        self.out_degree = 0


        self.out_dim = layers_list[0].out_dim
        self.n_of_input_layers = len(layers_list)

        self.last_layers_list = layers_list
        for layer in layers_list[1:]:
            if layer.out_dim != self.out_dim:
                raise ValueError("input layers out_dim not match")

        self.last_layer_list = layers_list
        if name:
            self.name = name+str(self.out_dim)+" : %d"%ADD.count
        else:
            self.name = "ADD"+str(self.out_dim)+" : %d"%ADD.count
        self.info_dic = {}
        ADD.count+=1


    def forward(self,layer_results):
        y = layer_results[0]
        for results in layer_results[1:]:
            y+=results
        return y

    def auto_forward(self,is_train):
        layer_results = [layer.info_dic['y'] for layer in self.last_layer_list ]
        self.info_dic['y'] = self.forward(layer_results)

    def backward(self,grid_on_y):
        # 只用处理单输出
        for layer in self.last_layers_list:
            self.info_dic['grid_on_%s'%layer.name] = grid_on_y
        # return grid_on_y

    def auto_backward(self,is_train):
        grid_on_y = self.next_layer_list[0].info_dic['grid_on_%s'%self.name]
        for layer in self.next_layer_list[1:]:
            grid_on_y+=layer.info_dic['grid_on_%s'%self.name]
        for layer in self.last_layers_list:
            self.info_dic['grid_on_%s'%layer.name] = grid_on_y
        


