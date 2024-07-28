import numpy as np
from pylab import *
from sklearn.metrics import r2_score

def polynomialRegression():
    np.random.seed(2)
    pageSpeeds = np.random.normal(3.0, 1.0, 1000)
    purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

    plt.scatter(pageSpeeds, purchaseAmount)
    plt.show()

    x = np.array(pageSpeeds)
    y = np.array(purchaseAmount)

    p4 = np.poly1d(np.polyfit(x, y, 4)) #4 is the degreee of the polynomial equation

    xp = np.linspace(0, 7, 100)
    plt.scatter(x, y)
    plt.plot(xp, p4(xp), c='r')
    plt.show()

    r2 = r2_score(y, p4(x))

    print(r2)   

polynomialRegression()