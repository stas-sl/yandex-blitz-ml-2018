import numpy as np

from interface import BlackBox


def estimate_grad(x, n, sigma):
    g = np.zeros_like(x).astype(float)
    for i in range(n):
        dx = np.random.normal(0, 1, x.shape)
        g += bbox.calc(x + sigma * dx) * dx
        g -= bbox.calc(x - sigma * dx) * dx

    return g / 2 / n / sigma


sigma = 0.1
n_estimate = 130
n_iter = 100000
lr = 3
reg = 0.0012

bbox = BlackBox()
x = bbox.get_image().astype(float)
original = x.copy()
for i in range(n_iter):
    grad = estimate_grad(x, n_estimate, sigma)
    grad -= reg * (x - original) / x.shape[0]
    x += lr * grad / np.linalg.norm(grad)
    x = np.clip(x, 0, 255)
    p = bbox.calc(x)
    if p > 0.5:
        break

print(*x.astype('uint8'))
