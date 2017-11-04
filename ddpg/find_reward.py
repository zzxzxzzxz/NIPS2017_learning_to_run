import sys

n = 15
idx = 0
q = [''] * n

if sys.argv[2] == 'max':
    target = 100
else:
    target = float(sys.argv[2])

M = -10
with open(sys.argv[1]) as f:
    episode = -1
    for line in f.readlines():
        q[idx] = line
        idx = (idx + 1) % n

        if line.startswith('reward:'):
            r = float(line.split()[1].strip(','))
            M = max(M, r)

            if r >= target:
                print(''.join(q[idx:] + q[:idx]), end='')
                break
    else:
        if sys.argv[2] == 'max':
            print('max: ', M)
        else:
            print('Not Found')
