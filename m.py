# run: python m.py < input/m1.in

import numpy as np

n, m = map(int, input().split())
a = np.array([list(map(int, input().split())) for i in range(m)], ndmin=2)
w = np.ones(n)
x = np.zeros((m, n))
if m >= 1:
    x[np.arange(m), a[:, 0] - 1] = 1
    x[np.arange(m), a[:, 1] - 1] = -1
v = np.zeros_like(w)
for i in range(1000):
    z = w @ x.T
    h = 1 / (1 + np.exp(-z))
    gradient = x.T @ (h - 1) / m
    v = 0.99 * v + gradient
    w -= v

print(*(w.argsort()[::-1] + 1))
