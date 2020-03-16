from .. import *

class Optimizer(object):
    """docstring for Loss"""
    TYPE = 'optimizer'
    def __init__(self, name,learning_rate):
        self.name = name
        self.learning_rate = learning_rate

        
    def add_parameters(self,all_pamaters):
        self.all_pamaters = all_pamaters

    def optimize(self):
        # print("do optimize")
        pass


class Sgd(Optimizer):
    def __init__(self, name="sgd",learning_rate = 1e-3):
        self.name = name
        self.learning_rate = learning_rate

    def optimize(self,t):
        # print("do optimize sgd")
        for pamater in self.all_pamaters:
            pamater.value -= (pamater.gradient+pamater.reg_gradient)*self.learning_rate
            # pamater.value_check()
        

class Adam(Optimizer):
    def __init__(self, name='adam',learning_rate = 1e-3,alpha = 0.9,beta=0.999,delta=1e-8,):
        self.name = name
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.s = {}# 一阶矩
        self.r = {}# 二阶矩

    
    def add_parameters(self,all_pamaters):
        learning_rate = self.learning_rate
        alpha = self.alpha
        beta = self.beta
        delta = self.delta

        self.all_pamaters = all_pamaters
        for pamater in all_pamaters:
            # print(type(pamater.value))
            # print(pamater.value.shape)
            # s = 
            # 这里用zeroslike会在cupy报错
            self.s[pamater.name] = np.zeros_like(pamater.value) # 一阶矩
            # r = 
            self.r[pamater.name] = np.zeros_like(pamater.value) # 二阶矩

    def optimize(self,t):
        learning_rate = self.learning_rate
        alpha = self.alpha
        beta = self.beta
        delta = self.delta
        for pamater in self.all_pamaters:
            self.s[pamater.name]*=alpha
            self.s[pamater.name]+=(1-alpha)*(pamater.gradient+pamater.reg_gradient)
            #self.s[pamater.name] = alpha*self.s[pamater.name] + (1-alpha)*(pamater.gradient+pamater.reg_gradient)
            self.r[pamater.name]*=beta
            self.r[pamater.name]+=(1-beta)*((pamater.gradient+pamater.reg_gradient)**2)
            #self.r[pamater.name] = beta*self.r[pamater.name] + (1-beta)*((pamater.gradient+pamater.reg_gradient)**2)
            s_ = self.s[pamater.name]/(1-alpha**t)
            r_ = self.r[pamater.name]/(1-beta**t)
            gradient = s_/(np.sqrt(r_)+delta)
            pamater.value-=gradient*learning_rate
            del gradient,s_,r_