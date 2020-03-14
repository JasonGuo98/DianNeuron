import pickle
import os
try:
    import cupy as cp
    import_CP = True
except:
    print("cann't use cupy")
import numpy as np


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y
    

def load_CIFAR10(ROOT):
    """load all cifar data"""
    xs = [] 
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)#使变成行向量
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    if import_CP:
        return cp.array(Xtr), cp.array(Ytr), cp.array(Xte), cp.array(Yte)
    else:
        return Xtr, Ytr, Xte, Yte