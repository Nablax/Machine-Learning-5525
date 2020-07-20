import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_score import score, true_values

# I don't quite understand what I need to do in this question
# so I realize gradient boosting (know the loss and true value)
# and guessing gradient (unknown loss and the true value)

# compute the gradient, the derivation of the loss function
def gradient(y_pred):
    grad = 1/score(y_pred) * (np.log2((true_values + 1)/(y_pred + 1))) / (y_pred + 1)
    return grad


# use the gradient as the results of some hypothesis
def gradient_boosting(learning_rate, epoch, true_len):
    H = []
    F = np.zeros((true_len, 1))
    H.append(F)
    l_r = learning_rate
    scores = []
    for i in range(epoch):
        y_pred = F
        sc = score(y_pred)
        if sc <= 0.001:
            break
        # print(sc)
        scores.append(sc)
        grad = gradient(y_pred)
        # regard the gradient as a result for a hypothesis
        F += l_r * grad
        H.append(grad)
    return scores


# random guess the gradient
def random_boosting(epoch, true_len):
    H = []
    F = np.zeros((true_len, 1))
    # for every single value, the upper bound to guess
    guess_range = np.zeros((true_len, 1)) + 10
    H.append(F)
    scores = []
    scores.append(score(F))
    for i in range(epoch):
        index_now = i % true_len
        h = np.zeros((true_len, 1))
        # generate random gradient within the bound in the guess_range
        rand_grad = np.random.uniform(0, guess_range[index_now])
        h[index_now] = rand_grad
        # regard the random gradient as result of a hypothesis, and add to F
        y_pred = F + h
        sc = score(y_pred)
        # If loss increase, don't add this gradient and set the upper bound to this random gradient
        if sc > scores[-1]:
            guess_range[index_now] = rand_grad
            sc = scores[-1]
        else:
            F = F + h
        scores.append(sc)
    return scores


if __name__ =='__main__':
    true_len = 21283
    epoch = 100000
    scores = gradient_boosting(0.1, epoch, true_len)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Gradient boosting')
    plt.plot(scores)
    plt.show()

    scores = random_boosting(epoch, true_len)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Gradient guessing')
    plt.plot(scores)
    plt.show()

