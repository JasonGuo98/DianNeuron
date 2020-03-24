from .. import *
import copy

class Loss(object):
    """docstring for Loss"""
    TYPE = "lossfunction"
    def __init__(self,name = "No_loss"):
        self.name = name

        pass

    def cal_loss_and_grid(self,y_pred,y_batch):
        return "loss_on_batch","grid_info_dic"

    def add_parameters(self,parameter_list):
        # 用于加入正则loss
        self.parameter_list = parameter_list

class CrossEntropy(Loss):
    """docstring for CrossEntropy"""
    def __init__(self, name = "CrossEntropy"):
        self.name = name

    def cal_loss_and_grid(self,y_pred,y_batch):
        delta = 1e-6
        # (np.min(y_pred))
        loss_on_batch = np.mean(-np.sum(y_batch*np.log(y_pred),axis = 1))# 由于forward的时候添加了delta，输出不会等于0或1
        #这里计算 从loss 回传的梯度
        # 从正则项回传的梯度由参数自己维护
        # print("loss_on_batch:",loss_on_batch)
        reg_loss = 0
        for parameter in self.parameter_list:
            reg_loss+=parameter.reg_loss
            # 增加正则项loss
        grid_on_y_pred = -y_batch/(y_pred)

        return loss_on_batch,reg_loss,grid_on_y_pred
    

class HingeLoss(Loss):
    """docstring for HingeLoss"""
    def __init__(self, name = "HingeLoss",margin = 1,p=1):
        self.name = name
        self.margin = margin
        self.p = p

    def cal_loss_and_grid(self,y_pred,y_batch):
        margin = self.margin# 默认为1
        p = self.p

        _y_pred = copy.deepcopy(y_pred)
        right_score = _y_pred[y_batch == 1]
        _y_pred[y_batch==1]-=margin
        grid_on_y_pred = np.max(0,(margin+_y_pred-right_score)**p)
        loss_on_batch = np.mean(np.max(grid_on_y_pred,axis=1))
        # 如果所有错误标签的取值+1都比正确标签的取值小，则loss等于0
        reg_loss = 0
        for parameter in self.parameter_list:
            reg_loss+=parameter.reg_loss
            # 增加正则项loss
        grid_on_y_pred[grid_on_y_pred > 0] = 1
        grid_on_y_pred[y_batch==1] = np.sum(grid_on_y_pred,axis=1)


        return loss_on_batch,reg_loss,grid_on_y_pred

class FocalLoss(Loss):
    def __init__(self,name='FocalLoss', alpha=0.25, gamma=2):
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        assert gamma != 0
    def cal_loss_and_grid(self,y_pred,y_batch):
        '''
        使用FocalLoss前一定要经过**softmax**激活函数
        '''
        log_pred = np.log(y_pred)
        pow_pred = np.power((1-y_pred)*y_batch,self.gamma)
        #forward
        batch_loss = -self.alpha * pow_pred *log_pred 
        loss = np.mean(np.sum(batch_loss,axis = 1))
        #backward
        grid_on_y_pred = -self.gamma * np.power((1-y_pred),self.gamma-1)*log_pred - pow_pred / y_pred

        reg_loss = 0
        for parameter in self.parameter_list:
            reg_loss+=parameter.reg_loss

        return loss,reg_loss,grid_on_y_pred