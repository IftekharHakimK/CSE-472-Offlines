import numpy as np

class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        self.params = params

    def g(self, z):
        """
        :param z:
        :return:
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        self.theta = np.zeros((X.shape[1], 1))
        # print(self.theta.shape)

        alpha = self.params['alpha']
        iterations = self.params['iterations']
        m = X.shape[0]
        for i in range(iterations):
            temp = alpha/m * (y - self.g(X.dot(self.theta)).T).dot(X)
            self.theta = self.theta + temp.T
        # print(self.theta)
        # print(self.theta.shape)

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # print("HERE        ", self.theta)
        temp = self.g(X.dot(self.theta))
        # print('temp ',temp)
        # print('temp shape ',temp.shape)
        y_pred = np.zeros((temp.shape[0], 1))
        for i in range(temp.shape[0]):
            if temp[i][0] > 0.5:
                y_pred[i][0] = 1
            else:
                y_pred[i][0] = 0
        # print('got ',y_pred)
        return y_pred
