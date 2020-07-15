import numpy as np

a = np.array([[1, 1],[2, 2]])
b = np.array([[0, 1],[2, 3]])
print(a*b)
print(np.dot(a, b))
print(np.sum(a*b))