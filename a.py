# run: python a.py

# Command line to generate random input
# {echo 100000; paste -d ' ' <(gshuf -r -i 0-1000000000 -n 100000) <(gshuf -r -i 1-1000 -n 100000) } > input/a.in

# Important!!! In this task you MUST read and write files instead of stdin/stdout
# otherwise you will receive PE (Presentation Error)
# Should replace input filename with 'stump.in' before submitting
# with open('stump.in') as f:
with open('input/a.in') as f:
    n = int(f.readline())
    x, y = zip(*sorted([list(map(int, f.readline().split())) for i in range(n)],
                       key=lambda x: x[0]))

s_left = 0
s_right = sum(y)
s2_left = 0
s2_right = sum(map(lambda x: x * x, y))

best_cost = None
best_a = y[0]
best_b = s_right / n
best_c = x[0] - 1

for i in range(n - 1):
    s_left += y[i]
    s_right -= y[i]
    s2_left += y[i] ** 2
    s2_right -= y[i] ** 2
    if x[i] != x[i + 1]:
        a = s_left / (i + 1)
        b = s_right / (n - i - 1)
        c = (x[i] + x[i + 1]) / 2
        cost_left = (i + 1) * a * a - 2 * a * s_left + s2_left
        cost_right = (n - i - 1) * b * b - 2 * b * s_right + s2_right
        f = cost_left + cost_right
        if best_cost is None or best_cost > f:
            best_cost = f
            best_a = a
            best_b = b
            best_c = c

# Should replace output filename with 'stump.out' before submitting
# with open('stump.out', 'w') as f:
with open('output/a.out', 'w') as f:
    print(best_a, best_b, best_c, file=f)
