# run: python o.py < input/o.in

import numpy as np

k, U, M, D, T = map(int, input().split())


def convert(s):
    s = s.split()
    return int(s[0]), int(s[1]), float(s[2])


train = [convert(input()) for i in range(D)]

factors = 20
epochs = 10
init_mean = 0
init_std = .1
lr = .01
reg = .02
ratings_mean = sum(x[2] for x in train) / len(train)

bu = np.zeros(U)
bi = np.zeros(M)
pu = np.random.normal(init_mean, init_std, (U, factors))
qi = np.random.normal(init_mean, init_std, (M, factors))

for e in range(epochs):
    for u, i, r in train:
        err = r - (ratings_mean + bu[u] + bi[i] + qi[i] @ pu[u])

        bu[u] += lr * (err - reg * bu[u])
        bi[i] += lr * (err - reg * bi[i])
        pu[u] += lr * (err * qi[i] - reg * pu[u])
        qi[i] += lr * (err * pu[u] - reg * qi[i])

for j in range(T):
    u, i = map(int, input().split())
    r = ratings_mean + bu[u] + bi[i] + qi[i] @ pu[u]
    r = max(2, min(k, r))
    print(r)
