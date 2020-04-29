from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt

class DecisionTreeRegression:
    def __init__(self):
        self.model = DTR()

    def train(self, x, y):
        self.model.fit(x, y)

    def test(self, x, y):
        accuracy = self.model.score(x, y)
        return accuracy

    def predict(self, x):
        y = self.model.predict(x)
        return y

    def graph(self,test_x,test_y):
        real = [0 for i in range(300)]
        cnt = [0 for i in range(300)]

        scaler = MinMaxScaler()
        scaler.fit([[7700, 21, 95], [0, 2, 34]])
        test_x = np.hstack((test_x[:,:-3], scaler.inverse_transform(test_x[:,-3:])))
        for row in range(len(test_x)):
            if test_x[row][3]==1 and test_x[row][6]==1 and test_x[row][10]==1 and test_x[row][14]==1:
                if int(test_x[row][-3])>=300:
                    cnt[299] += 1
                    real[299] += test_y[row]
                else:
                    cnt[int(test_x[row][-3])] += 1
                    real[int(test_x[row][-3])] += test_y[row]

        #real value dots in graph
        real_x = []
        real_y = []
        for i in range(300):
            if cnt[i]!=0:
                real_x.append(i)
                real_y.append(real[i]/cnt[i])
        print(real_x)

        i = [i for i in range(300)]
        x = np.array([[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, i, 20, 60] for i in range(300)])

        # scaling (0,1)
        x = np.hstack((x[:, :-3], scaler.transform(x[:, -3:])))

        y = self.predict(x)
        plt.scatter(real_x, real_y, edgecolor="black",c="darkorange")
        plt.plot(i, y)
        plt.show()