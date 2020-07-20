import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_error_rate import error_rate


# the algorithm is written in the report
def random_guessing_proportion(epoch, true_len):
    F = np.zeros((true_len, 1))
    one_num = int(error_rate(F)*true_len)
    h = np.zeros((true_len, 1))
    h[: one_num] = 1
    errs = []
    for i in range(epoch):
        np.random.shuffle(h)
        err_r = error_rate(h)
        h[h == 0] = -1
        if err_r < 0.5:
            F += (1 - err_r) * h
        else:
            F -= err_r * h # here because h is {-1, 1} so F + err*(-h)
        h[h == -1] = 0
        res = np.where(F > 0, 1, 0)
        errs.append(error_rate(res))
    return errs


if __name__ == '__main__':
    true_len = 21283
    epoch = 10000
    scores = random_guessing_proportion(epoch, true_len)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Guessing')
    plt.plot(scores)
    plt.show()




