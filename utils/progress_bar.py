from __future__ import division
import sys
import numpy as np

class ProgressBar(object):
    def __init__(self, total_steps, length=50):
        self.total_steps = total_steps
        self.length = int(length)

    def update(self, step):
        sys.stdout.write(' '*(min(self.length+14, 79)) + '\r')
        sys.stdout.flush()
        length = int(np.floor(step / self.total_steps * self.length))
        rest = self.length - length
        process = '[' + '='*length + ('>' if rest else '') + '.'*rest + ']'
        process += ' {:d}/{:d}\r'.format(step, self.total_steps)
        sys.stdout.write(process)
        sys.stdout.flush()

if __name__ == '__main__':
    import time
    pb = ProgressBar(100)
    for step in range(100):
        pb.update(step+1)
        time.sleep(0.1)
