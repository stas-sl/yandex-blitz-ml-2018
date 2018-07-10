# run: python n.py < input/n.in > output/n.out

n = int(input())

a = [list(map(int, input().split())) for i in range(n)]
for i in range(n):
    alpha = a[i][1] + 1
    beta = a[i][0] - a[i][1] + 1
    a[i].append(alpha / (alpha + beta))
    a[i].append(i)

a = sorted(a, key=lambda x: x[2])

print(*[x[-1] for x in a], sep='\n')

