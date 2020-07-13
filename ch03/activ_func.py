import numpy as np
import matplotlib.pyplot as plt
import math


# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.linspace(-4, 4, 100)
print(x)
sig = sigmoid(x)
plt.plot(x, sig)
plt.show()
