from data_handler import bagging_sampler
import numpy as np
import copy

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        estimators = []
        # make copy of base_estimator
        for i in range(n_estimator):
            estimators.append(copy.deepcopy(base_estimator))
        self.estimators = estimators
        self.n_estimator = n_estimator

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        for i in range(self.n_estimator):
            tX, ty = bagging_sampler(X, y)
            self.estimators[i].fit(tX, ty)

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        votes = np.zeros((X.shape[0], 1))
        
        for i in range(self.n_estimator):
            votes += self.estimators[i].predict(X)
        y_pred = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            if votes[i][0] > self.n_estimator/2:
                y_pred[i][0] = 1
            else:
                y_pred[i][0] = 0
        # print(votes)
        return y_pred

