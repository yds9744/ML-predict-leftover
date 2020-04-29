from sklearn.linear_model import LinearRegression as LR
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
        x = [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, i, 20, 60] for i in range(300)]
        y = self.predict(x)
        plt.plot(i, y)
        plt.show()