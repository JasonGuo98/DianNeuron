from .. import *

from ..utils.mini_batch_iter import MINI_BATCH_ITER
from .classifier import Classifier
from .nn_model import NN_Model

# def check_attributes(check_attributes):
#     def func(*args,**kwargs):


#     try check_attributes:
#         if check_attributes:
#             func()
#     except Exception as e:
#         raise
#     else:
#         pass
#     finally:
#         pass

class Perception(Classifier):
    """
    简单的多层感知机实现
    """
    def __init__(self,input_dim,n_of_classes,*args,**kwds):
        if type(input_dim) != int or type(n_of_classes) != int:
            raise TypeError("input_dim and n_of_classes must be int")
        super().__init__(input_dim,n_of_classes)
        # 初始化后要求执行build函数，来构建模型
        self.model = None


    def build(self,input_layer,output_layer,optimizer = "sgd",learning_rate = 1e-3,\
                                    lossfunction = "CrossEntropy"):
        if input_layer.out_dim != self.input_dim:
            raise ValueError("input_layer dim is ",input_layer.out_dim,\
                " model input_dim must be",self.input_dim)
        if output_layer.out_dim != self.n_of_classes:
            raise ValueError("output_layer dim is ",output_layer.out_dim,\
                " model output_dim must be",self.n_of_classes)
        
        self.model = NN_Model(input_layer = input_layer,output_layer = output_layer, \
                    optimizer = optimizer,learning_rate = learning_rate,lossfunction = lossfunction)

        

    def train_on_batch(self,x_batch,y_batch):
        loss_on_batch,reg_loss_on_batch,grid_on_input = self.model.train_on_batch(x_batch,y_batch)
        return loss_on_batch,reg_loss_on_batch # ,reg_loss_on_batch,loss_on_input

    # @check_attributes(self.model)
    def train(self,X_train,y_train,test_set = None,batch_size = 64,epoch = 1,\
                        epoch_shuffle = True):
        # 把每个样本的特征reshape成一个向量    
        # X_train = np.reshape(X_train,(X_train.shape[0],-1))
        
        loss_list = []
        train_acc = []
        test_acc = []
        history = {}
        history['test_acc'] = test_acc
        history['train_acc'] = train_acc
        history['loss_list'] = loss_list
        if test_set:
            X_test = test_set[0]
            y_test = test_set[1]
        epoch_count = 0
        for ep in range(epoch):
            print("train on epoch %d:"%epoch_count)
            epoch_count +=1
            batch_iter = iter(MINI_BATCH_ITER(X_train,y_train,\
                    shuffle = epoch_shuffle,batch_size=batch_size))
            # build mini-batch iter
            for x_batch,y_batch in batch_iter:
                loss_on_batch,reg_loss_on_batch = self.train_on_batch(x_batch,y_batch)
                loss_list.append(loss_on_batch)
                print("\rloss now:%.4f,reg_loss_on_batch:%.4f,sum loss: %.4f"%(loss_on_batch,reg_loss_on_batch,loss_on_batch+reg_loss_on_batch/batch_size),end='')
                del x_batch,y_batch
            train_acc.append(self.test(X_train,y_train,put_out = False))
            if test_set:
                test_acc.append(self.test(X_test,y_test,put_out = False))
                print("\n Train ACC = %.2f %%, Test ACC =  %.2f %%"%(100*train_acc[-1],\
                            100*test_acc[-1]))
            else:
                print("\n Train ACC = %.2f %%"%(100*train_acc[-1]))

        print("==> finish training process")

        return history
        
    def test(self,X_test,y_test,put_out = True):
        """直接调用predict进行测试"""
        n_of_test = X_test.shape[0]
        y_pred = np.argmax(self.model.forward(X_test,is_train = False),axis=1)
        acc = sum((y_pred)==np.argmax(y_test,axis=1))/n_of_test
        if put_out:
            print("\n==> ACC = %.2f %%"%(100*acc))
        return acc
    
    def predict(self,X_test,one_hot = False):
        n_of_test = X_test.shape[0]
        y_pred = self.model.forward(X_test)
        if one_hot == False:
            return np.argmax(y_pred,axis = 1)        
        return y_pred
        
