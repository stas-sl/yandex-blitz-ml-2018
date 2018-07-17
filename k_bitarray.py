# run: python k_bitarray.py < input/k.in

n = int(input())
m = 200000
a = bytearray(m)
for i in range(n):
    h = hash(input()) % (m * 8)
    a[h // 8] |= (1 << h % 8)

cnt = 0
for h in a:
    while h > 0:
        if h & 1:
            cnt += 1
        h >>= 1

print(cnt)
