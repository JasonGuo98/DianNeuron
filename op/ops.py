
class OP(object):
	"""docstring for OP"""
	TYPE = "op"
	def __init__(self, arg):
		super(OP, self).__init__()
		

class ADD(object):
 	"""docstring for ADD"""
    count = 0
 	def __init__(self, layers_list,name):
        # out_dim check
        self.out_dim = layers_list[0].out_dim
        self.n_of_input_layers = len(layers_list)
        self.input_layers_list = layers_list
        for layer in layers_list[1:]:
            if layer.out_dim != self.out_dim:
                raise ValueError("input layers out_dim not match")

 		self.last_layer_list = layers_list
        if name:
            self.name = name+str(self.out_dim)+" : %d"%ADD.count
        else:
            self.name = "ADD"+str(self.out_dim)+" : %d"%ADD.count

        ADD.count+=1


 	def forward(self,layer_results):
 		# todo 完成求和算子
        y = layer_results[0]
        for results in layer_results[1:]:
            y+=results
 		return y

 	def backword(self,grid_info_dic):
		grid_on_y = grid_info_dicc['grid_on_y']
		grid_on_inputs = [grid_on_y for i in range(self.n_of_input_layers)]
		return grid_on_inputs


def get_op(opname):
    return OP()