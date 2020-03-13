import numpy as np
from classifier import *
class KNN(Classifier):
    def __init__(self,input_dim,n_of_classes,K=1,distance_type = "L2",weight = "uniform"):
        super().__init__(input_dim,n_of_classes)
        if distance_type not in ["L1","L2","COSINE"]:
            raise ValueError
        if weight not in ["uniform","distance"]:
            raise ValueError
        self.K = K
        self.distance_type = distance_type
        self.weight = weight
        
    def train(self,X_train,y_train):
        # 把每个样本的特征reshape成一个向量
        self.X_train = np.reshape(X_train,(X_train.shape[0],-1))
        self.y_train = y_train
        print("==> finish training process")
        
    def test(self,X_test,y_test):
        """直接调用predict进行测试"""
        n_of_test = X_test.shape[0]
        _X_test = np.reshape(X_test,(n_of_test,-1))
        results = []
        for i in range(n_of_test):
            pred = self.predict(_X_test[i])
            results.append(pred)
        print("==> ACC = %.2f %%"%(100*sum(np.array(results)==y_test)/len(y_test)))
    
    def predict(self,input_img):
        """预测单个样本"""
        _img = np.reshape(input_img,-1)
        if self.distance_type == "L1":
            dis = np.sum(np.abs((self.X_train-_img)),axis=1)
        elif self.distance_type == "L2":
            dis = np.sqrt(np.sum(np.power((self.X_train-_img),2),axis=1))
        elif self.distance_type == "COSINE":
            dis = 1-self.X_train.dot(_img)/np.sqrt(np.sum(self.X_train**2,axis=1)*np.sum(_img**2))
        else:
            raise ValueError
        sorted_arg = arg_minK(dis,k=self.K)
        if self.weight == "uniform":
            vote = [0 for i in range(self.n_of_classes)]
            for i in range(self.K):
#                 print("dist = ",(dis[sorted_arg[i]]+1e-6))
                vote[self.y_train[sorted_arg[i]]] += 1
            result = randomArgMax(vote)
        elif self.weight == "distance":
            vote = [0 for i in range(self.n_of_classes)]
            for i in range(self.K):
#                 print("dist = ",(dis[sorted_arg[i]]+1e-6))
                vote[self.y_train[sorted_arg[i]]] += 1/(dis[sorted_arg[i]]+1e-6)
            result = randomArgMax(vote)
        else:
            raise ValueError
        
        return result
        
    def cal_dist(self,X_test):
        """利用numpy的广播操作，减少循环次数,计算X_test 所有样本和X_train 所有样本的距离"""
        n_of_test = X_test.shape[0]
        n_of_train = self.X_train.shape[0]
        _X_test = np.reshape(X_test,(n_of_test,-1))
        """
        对于X_test 维度为 A*F, X_train 维度为 B*F
        
        距离矩阵dist 中， i,j 号元素表示X_test[i] 与 X_train[j]的距离
        
        则对于 L2 距离 dist[i][j] = sum(X_test[i]**2) + sum(X_train[j]**2) - 2 * X_test[i] * X_train[j].T (行向量乘以列向量)
        
        对于L1距离, 不能这么做
        """
        n_of_train = self.X_train.shape[0]
        if self.distance_type == "L1":
            dist = np.zeros((n_of_test,n_of_train))
            for i in range(n_of_test):
                _ = np.sum(np.abs(self.X_train - _X_test[i]),axis=1)
                dist[i] = _
        elif self.distance_type == "L2":
            dist = -2*_X_test.dot(self.X_train.T) +\
            np.sum(np.square(_X_test),axis=1,keepdims=1) +\
            np.sum(np.square(self.X_train),axis=1).T
        elif self.distance_type == "COSINE":
            dist = 1-_X_test.dot(self.X_train.T)/\
            np.sqrt(np.sum(_X_test**2,axis=1,keepdims=True) * np.sum(self.X_train**2,axis=1).T)
        else:
            del _X_test
            raise ValueError
        del _X_test
        
        return dist
        
    def fast_test(self,X_test,y_test):
        n_of_test = X_test.shape[0]
        results = []
        
        dist = self.cal_dist(X_test)
        
        
        if self.weight == "uniform":
            for i in range(n_of_test):
                sorted_arg = arg_minK(dist[i],k=self.K)
                vote = [0 for i in range(self.n_of_classes)]
                for j in range(self.K):
                    vote[self.y_train[sorted_arg[j]]] += 1
                result = randomArgMax(vote)

                results.append(result)
        elif self.weight == "distance":
            for i in range(n_of_test):
                sorted_arg = arg_minK(dist[i],k=self.K)
                vote = [0 for i in range(self.n_of_classes)]
                for j in range(self.K):
                    vote[self.y_train[sorted_arg[j]]] += 1/(dist[i][sorted_arg[j]]+1e-6)
                result = randomArgMax(vote)

                results.append(result)
        else:
            raise ValueError
                
        print("==> ACC = %.2f %%"%(100*sum(np.array(results)==y_test)/len(y_test)))
        return sum(np.array(results)==y_test)/len(y_test)
        
    def fast_test_with_dist(self,dist,y_test):
        
        n_of_test = dist.shape[0]
        results = []
        
        if self.weight == "uniform":
            for i in range(n_of_test):
                sorted_arg = arg_minK(dist[i],k=self.K)
                vote = [0 for i in range(self.n_of_classes)]
                for j in range(self.K):
                    vote[self.y_train[sorted_arg[j]]] += 1
                result = randomArgMax(vote)

                results.append(result)
        elif self.weight == "distance":
            for i in range(n_of_test):
                sorted_arg = arg_minK(dist[i],k=self.K)
                vote = [0 for i in range(self.n_of_classes)]
                for j in range(self.K):
                    vote[self.y_train[sorted_arg[j]]] += 1/(dist[i][sorted_arg[j]]+1e-6)
                result = randomArgMax(vote)

                results.append(result)
        else:
            raise ValueError
            
            
        
        
        print("==> ACC = %.2f %%"%(100*sum(np.array(results)==y_test)/len(y_test)))
        return sum(np.array(results)==y_test)/len(y_test)
        
    def fast_predcit(self,X_test,y_test):
        """预测多个样本"""
        n_of_test = X_test.shape[0]
        results = []
        dist = self.cal_dist(X_test)
        
        for i in range(n_of_test):
            sorted_arg = arg_minK(dist[i],k=self.K)
            vote = [0 for i in range(self.n_of_classes)]
            for j in range(self.K):
                vote[self.y_train[sorted_arg[j]]] += 1
            result = randomArgMax(vote)
            
            results.append(result)
        return results