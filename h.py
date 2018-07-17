# run: python h.py < input/h.in

import math

n = int(input())
for i in range(n):
    r, d = map(float, input().split())
    if r == -1:
        r = 7.9452149399471397  # mean rating
    r = math.exp(r)
    r = (r - 4087.8388613559214) / 3384.0914139209362
    d = math.log(d + 0.0001)
    d = (d + 4.5436589133677519) / 1.5420236847938951
    s = 0.72500236 * r - 1.94289503 * d
    print(s)
