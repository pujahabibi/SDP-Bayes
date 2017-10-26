__author__ = 'Kiki Rizki Arpiandi'
import numpy as np
import random
import matplotlib.pyplot as plt


class model:
    def __init__(self, x,y):
        self.y = y
        self.x = x
        self.theta0 = random.randrange(-5, 5, 1)
        self.theta1 = random.randrange(-5, 5, 1)

    def prediksi(self, x):
        return (self.theta1 * x) + self.theta0

    def Error(self):
        return np.average((self.prediksi(self.x) - self.y) ** 2) / 2

    def delta_J_delta_theta0(self):
        return np.average((self.prediksi(self.x) - self.y))

    def delta_J_delta_theta1(self):
        return np.average((self.prediksi(self.x) - self.y) * self.x)

    def plot(self):
        plt.plot(self.x, reg.prediksi(self.x))
        plt.plot(self.x, self.y, 'ro')
        plt.show()

    def do_gradient_descent(self):
        error = 0
        while (abs(reg.Error() - error) > 0.0000001):
            error = reg.Error()
            temp0 = self.theta0 - 0.01 * reg.delta_J_delta_theta0()
            temp1 = self.theta1 - 0.01 * reg.delta_J_delta_theta1()
            self.theta0 = temp0
            self.theta1 = temp1


data_x = np.array([0., 3., 5., 6., 9.])
data_y = np.array([72., 95., 112., 77., 54.])
reg = model(data_x,data_y)
reg.do_gradient_descent()
reg.plot()
print(reg.theta1,reg.theta0)