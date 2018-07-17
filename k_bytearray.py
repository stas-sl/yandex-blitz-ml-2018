# run: python k_bytearray.py < input/k.in

n = int(input())
m = 1000000
a = bytearray(m)
for i in range(n):
    a[hash(input()) % m] = 1

print(a.count(1))
