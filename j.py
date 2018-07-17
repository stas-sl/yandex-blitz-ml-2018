# run: python j.py < input/j1.in

import numpy as np

n, m = map(int, input().split())
a = np.array([list(map(float, input().split())) for i in range(n)])
x = a[:, :-1]
y = a[:, -1]
w = np.zeros(m)
lr = 0.01
for i in range(100):
    y_hat = np.sign(w @ x.T)
    w += lr * x.T @ (y - y_hat)
print(*w)
