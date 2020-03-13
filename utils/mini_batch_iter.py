import numpy as np
import copy
class MINI_BATCH_ITER(object):
  def __init__(self,X,y,shuffle=True,batch_size = 64):
    if len(X) != len(y):
      raise ValueError("X size is %d y size is %d"%(len(X),len(y)))
    if shuffle:
      permutation = np.random.permutation(len(X))
      self.X = copy.deepcopy(X[permutation])
      self.y = copy.deepcopy(y[permutation])
    else:
      self.X = copy.deepcopy(X)
      self.y = copy.deepcopy(y)
    self.start_index = 0
    self.data_set_size = len(X)
    self.batch_size = batch_size
    

  def __iter__(self):
    return self
 
  def __next__(self):
    end_index = self.start_index+self.batch_size
    if end_index <= self.data_set_size:
      batch_X = self.X[self.start_index:end_index]
      batch_y = self.y[self.start_index:end_index]
      self.start_index+=self.batch_size 
      return batch_X,batch_y
    else:
      raise StopIteration