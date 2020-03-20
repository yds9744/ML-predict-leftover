from sklearn.linear_model import LinearRegression as LR

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