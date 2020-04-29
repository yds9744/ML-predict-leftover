from sklearn.neighbors import KNeighborsRegressor as KR
from matplotlib import pyplot as plt

class KnnRegression:
    def __init__(self):
        self.model = KR(n_neighbors=30)

    def train(self, x, y):
        self.model.fit(x, y)

    def test(self, x, y):
        accuracy = self.model.score(x, y)
        return accuracy

    def predict(self, x):
        y = self.model.predict(x)
        return y

    def graph(self):
        i = [i for i in range(300)]
        x = [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, i, 19, 40] for i in range(300)]
        y = self.predict(x)
        plt.plot(i, y)
        plt.show()
