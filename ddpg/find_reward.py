import sys

n = 15
idx = 0
q = [''] * n

with open(sys.argv[1]) as f:
    episode = -1
    for line in f.readlines():
        q[idx] = line
        idx = (idx + 1) % n

        if line.startswith('reward:'):
            r = float(line.split()[1].strip(','))

            if r >= float(sys.argv[2]):
                print(''.join(q[idx:] + q[:idx]), end='')
                break
    else:
        print('Not Found')
