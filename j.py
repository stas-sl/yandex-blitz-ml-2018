# run: python j.py < input/j1.in

import numpy as np

n, m = map(int, input().split())
a = np.array([list(map(float, input().split())) for i in range(n)])
x = a[:, :-1]
y = a[:, -1] > 0
w = np.zeros(m)
lr = 0.1
for i in range(100):
    z = np.dot(x, w)
    h = 1 / (1 + np.exp(-z))
    gradient = x.T @ (h - y) / n
    w -= lr * gradient

print(*w)
