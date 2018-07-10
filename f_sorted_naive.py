# run: python f_sorted_naive.py < input/f5.in

# Command line to generate random input
# {echo 100000; paste -d ' ' <(gshuf -r -i 0-1000 -n 100000) <(gshuf -r -i 0-1000 -n 100000) } > input/f5.in

# Interestingly, this is the only python solution that I managed
# to squeeze into 3s time limit. Actually the longest test case took 2.995s
# so I was very lucky :)
# It uses sorted list to store and count points left to the current.
# Though search in this list takes O(log N), deletion items from it
# still takes O(N), so worst case is O(N*N).
# I tried to implement binary tree as in f_fastest.cpp, but surprisingly
# python version of it was even slower than this version and got TL

from bisect import bisect_left, bisect_right

n = int(input())
t, y = zip(*sorted(tuple(map(float, input().split())) for i in range(n)))
y_sorted = sorted(y)

nom = denom = 0
i = n - 1

while i >= 0:
    j = i
    while j >= 0 and t[i] == t[j]:
        l = bisect_left(y_sorted, y[j])
        y_sorted.pop(l)
        j -= 1

    for k in range(j + 1, i + 1):
        l = bisect_left(y_sorted, y[k])
        r = bisect_right(y_sorted, y[k])
        nom += l + (r - l) / 2
        denom += j + 1

    i = j

print(nom / denom)
