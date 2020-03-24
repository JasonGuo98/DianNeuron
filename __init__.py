
"""A simple neural network framework implemented manually using python"""


USING_CUPY = False
USING_NUMPY = False
# try:
#     import cupy as np
#     print("import cupy as np")
#     USING_CUPY = True
# except:
#     import numpy as np
#     print("import numpy as np")
#     USING_NUMPY = True
import numpy as np
print("import numpy as np")
USING_NUMPY = True

import DianNeuron.layer.layers as layers
import DianNeuron.op.ops as ops
import DianNeuron.model as model
import DianNeuron.utils as utils
import DianNeuron.parameter as parameter
import DianNeuron.loss as loss
import DianNeuron.activation as activation



__all__ = ["np","USING_NUMPY","USING_CUPY","layers",\
                "ops","model","utils","loss","parameter","activation"]

def set_random_seed(seed):
    if USING_CUPY:
        np.random.seed(seed)
        pass
    if USING_NUMPY:
        np.random.seed(seed)


