from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt

class LinearRegression:
    def __init__(self):
        self.model = LR(fit_intercept=True, normalize=True)

    def train(self, x, y):
        self.model.fit(x, y)

    def test(self, x, y):
        accuracy = self.model.score(x, y)
        return accuracy

    def predict(self, x):
        y = self.model.predict(x)
        return y

    def graph(self):
        print('Coefficients: \n', self.model.coef_)

        i = [i for i in range(300)]
        x = np.array([[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, i, 20, 60] for i in range(300)])

        # scaling (0,1)
        scaler = MinMaxScaler()
        scaler.fit([[7700, 21, 95], [0, 2, 34]])
        x = np.hstack((x[:, :-3], scaler.transform(x[:, -3:])))

        y = self.predict(x)
        plt.plot(i, y)
        plt.show()