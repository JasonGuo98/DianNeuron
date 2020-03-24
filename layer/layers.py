from .layers_class import *



def dense(out_dim,activation=None,\
    W_regularization="L2",W_regularizationRate=0.01,W_init='normal',b_init='zero',dtype = "float32",name = None):
    def inner_build(last_layer):
        dense_layer = Dense(last_layer, out_dim, activation, W_regularization, \
                            W_regularizationRate, W_init, b_init, dtype, name)
        return dense_layer

    return inner_build


def inputs(in_dim, name="Inputs"):
    layer = Inputs(in_dim, name)
    return layer

def dropout(keep_prob = 0.5,scale_train = True,name = "Dropout"):
    def inner_build(last_layer):
        dropout = Dropout(last_layer,keep_prob,scale_train,name)
        return dropout
    return inner_build

