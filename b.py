# run: python b.py < input/b.in

import sys
import numpy as np

from scipy.optimize import minimize

p = np.loadtxt(sys.stdin, dtype=[('x', 'float'), ('y', 'float')], ndmin=1)
x = p['x']
y = p['y']


def f(w, x, y):
    return (((w[0] * np.sin(x) + w[1] * np.log(x)) ** 2 + w[2] * x * x - y) ** 2).sum()


res = minimize(f, [1, 1, 1], (x, y), method='Nelder-Mead')
print(*res.x)
