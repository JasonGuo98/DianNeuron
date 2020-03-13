import numpy as np
MAX_N = 100

init_list = ["zero","normal","Xavier"]

class Parameter(object):
    count = 0
    def __init__(self,shape,name=None,regularization = None,regularizationRate = 0.0,init="zero",*args,**kwds):
        # print(shape,name,init)
        if type(shape) is not tuple:
            raise TypeError("shape must be a tuple")
        if init not in init_list:
            raise ValueError("init way must in: ",init_list)

        if regularization not in [None,"L1","L2"]:
            raise ValueError("regularization must be None, L1 or L2")
        self.regularization = regularization
        if type(regularizationRate) != float:
            raise TypeError("regularizationRate must be float")
        self.regularizationRate = regularizationRate

        if init == "zero":
            self.value = np.zeros(shape)
        elif init == "normal":
            self.value = np.random.standard_normal(shape)
        elif init == "Xavier":
            """正确性未知"""
            print("!!!Xavier正确性未知!!!")
            self.value=np.random.standard_normal(shape) * np.sqrt(1 / shape[0])
        if name:
            self.name = name+str(shape)+" : %d"%Parameter.count
        else:
            self.name = "Parameter"+str(shape)+" : %d"%Parameter.count
        self.gradient = np.zeros_like(self.value)
        self.reg_gradient = np.zeros_like(self.value)
        Parameter.count+=1

    def cal_regularization(self,):
        self.gradient = 0 # 梯度清零
        if self.regularization:
            if self.regularization=="L1":
                self.reg_loss = self.regularizationRate * np.sum(np.abs(self.value))
                ones = np.ones_like(self.value)
                ones[self.value<0] = -1
                self.reg_gradient=self.regularizationRate * ones
                pass
            elif self.regularization=="L2":
                self.reg_loss = self.regularizationRate * np.sum((self.value)**2)*0.5
                self.reg_gradient = self.regularizationRate * self.value
            else:
                self.reg_loss = 0
                self.reg_gradient = 0
        else:
            self.reg_loss = 0
            self.reg_gradient = 0
            
    def value_check(self,):
        print("do value check")
        self.value[self.value>MAX_N] = MAX_N
        self.value[self.value<-MAX_N] = -MAX_N

