from .layers_class import *


def dense(out_dim,activation=None,\
    W_regularization="L2",W_regularizationRate=0.01,W_init='normal',b_init='zero',name = None):
    def inner_build(last_layer):
        dense_layer = Dense(last_layer,out_dim,activation,W_regularization,\
            W_regularizationRate,W_init,b_init,name)
        return dense_layer
    return inner_build

def inputs(in_dim,name = "Inputs"):
    layer = Inputs(in_dim,name)
    return layer


