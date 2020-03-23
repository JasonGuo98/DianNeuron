from .. import *
import copy
import random

BATCH_SIZE = 2
INPUT_FEATURE = 4
DELTA = 1e-7

def activation_grid_check(activation,batch_size = BATCH_SIZE,input_feature = INPUT_FEATURE,delta = DELTA):
    """
    usage:
    activation = get_activation("ELU")
    activation_grid_check(activation)

    grad_numerical: -0.0020422267609648825
    grad_analytic: -0.002042226758153291
    ==> relative error on activation:ELU = 0.006884*1e-7
    """

    wxb = np.random.randn(batch_size,input_feature)
    ix = tuple([random.randrange(m) for m in wxb.shape])
    y = activation.forward(wxb)
    grid_on_y = np.zeros_like(wxb)
    grid_on_y[ix] = -1
    info_dic = {'y':y,"wxb":wxb}
    grid_on_wxb_analytic = activation.backward(grid_on_y,info_dic)


    wxb[ix] += delta
    y_plus_delta = activation.forward(wxb)
    wxb[ix] -= delta # reset

    wxb[ix] -= delta
    y_minus_delta = activation.forward(wxb)
    wxb[ix] += delta # reset

    grad_numerical = (grid_on_y*(y_plus_delta - y_minus_delta) / (2 * delta))[ix]
    print("grad_numerical:",grad_numerical)
    grad_analytic = grid_on_wxb_analytic[ix]
    print("grad_analytic:",grad_analytic)
    if abs(grad_numerical - grad_analytic) == 0:
        # 处理除零错误
        rel_error = 0
    else:
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))

    print("==> relative error on activation:%s = %lf*1e-7"%(activation.name,rel_error*1e7))


def dense_grid_check(layer,batch_size = BATCH_SIZE,delta = DELTA):
    """
    usage:
    input_layer = layers.inputs(10)
    layer1 = layers.dense(10,activation='tanh',W_regularization="L2",
                      W_regularizationRate=.01,W_init = 'kaiming_normal',b_init = 'zero')(input_layer)
    dense_grid_check(layer1)

    will print:
    grad_numerical: -0.05498857036378979
    grad_analytic: -0.05498857107816127
    ==> relative error on layer:Dense10 : 0 = 0.064956*1e-7
    """
    input_feature = layer.in_dim
    x = np.random.randn(batch_size,input_feature)
    ix = tuple([random.randrange(m) for m in x.shape])
    y = layer.forward(x)
    grid_on_y = np.zeros_like(x)
    grid_on_y[ix] = -1
    grid_on_x_analytic = layer.backward(grid_on_y)


    x[ix] += delta
    y_plus_delta = layer.forward(x)
    x[ix] -= delta # reset

    x[ix] -= delta
    y_minus_delta = layer.forward(x)
    x[ix] += delta # reset

    grad_numerical = (grid_on_y*(y_plus_delta - y_minus_delta) / (2 * delta))[ix]
    print("grad_numerical:",grad_numerical)
    grad_analytic = grid_on_x_analytic[ix]
    print("grad_analytic:",grad_analytic)
    if abs(grad_numerical - grad_analytic) == 0:
        # 处理除零错误
        rel_error = 0
    else:
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))

    print("==> relative error on layer:%s = %lf*1e-7"%(layer.name,rel_error*1e7))
