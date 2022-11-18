import numpy as np

class SearchSpace(object):
    def reset(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def addData(self ,data ,score):
        return

class KalmanFiltering(SearchSpace):
    def __init__(self ,bnd=[1,1,10]):
        self.bnd = bnd
        self.reset()

    def sample(self ,n =10):
        return np.random.multivariate_normal(self.mean ,self.cov ,size=n)

    def addData(self ,data ,score):
        score = score.clip(min = 1e-5)
        self.data = np.concatenate((self.data ,data))
        self.score = np.concatenate((self.score ,score))
        self.mean = np.average(self.data ,weights = self.score ,axis = 0)
        self.cov = np.cov(self.data.T ,ddof = 0 ,aweights= self.score)

    def reset(self):
        self.mean = np.zeros(len(self.bnd))
        self.cov = np.diag(self.bnd)
        if len(self.bnd) == 2:
            self.data =  np.array([[] ,[]]).T
        else:
            self.data = np.array([[] ,[] ,[]]).T
        self.score = np.array([])