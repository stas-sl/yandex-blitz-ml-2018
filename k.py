# run: python k.py < input/k.in

n = int(input())
p = 1000003
a = bytearray(p)
for i in range(n):
    a[hash(input()) % p] = 1

print(a.count(1))
