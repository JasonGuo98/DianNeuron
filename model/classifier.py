class Classifier(object):
    def __init__(self,input_dim,n_of_classes):
        self.feature_dim = input_dim
        self.input_dim = input_dim
        self.n_of_classes = n_of_classes
    def train(self):
    	
        return 
    def predict(self):
        return
    def test(self):
        return 
    def reset_hyperparameter(self,**hyperparameters):
        for k,v in hyperparameters.items():
            if k in vars(self) and type(v) == type(vars(self)[k]):
                vars(self)[k] = v
        pass