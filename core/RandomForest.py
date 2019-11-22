from core.common import *
from core.DecisionTree import *

class RandomForest(object):
    def __init__(self, max_depth, n_estimator, n_feature, train, label, sample_method='booststrap'):
        self.max_depth = max_depth
        self.n_estimator = n_estimator
        self.n_feature = n_feature
        self.train = train
        self.label = label
        self.sample_method = sample_method

        self.sub_train_set


    def build(self):
        for i in range(self.n_estimator):

