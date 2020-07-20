from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# process data, transfer {0, 1} label to {-1, 1}
def data_processing():
    train_data_all = np.genfromtxt('cancer_train.csv', delimiter=',')[1:, :]
    test_data_all = np.genfromtxt('cancer_test.csv', delimiter=',')[1:, :]
    train_data = train_data_all[:, :-1]
    train_label = train_data_all[:, -1]
    test_data = test_data_all[:, :-1]
    test_label = test_data_all[:, -1]
    train_label[train_label == 0] = -1
    test_label[test_label == 0] = -1
    return train_data, train_label, test_data, test_label


# compute error rate
def error_rate(y, y_pred):
    num_preds = len(y_pred)
    num_true_vals = len(y)
    if num_preds != num_true_vals:
        return 0
    val = np.sum(y - y_pred != 0) / num_true_vals
    return round(val, ndigits=5)


# adaboost predict
def adaboost_predict(test_data, test_label, H, alphas, iter=100):
    y_pred = np.zeros(test_data.shape[0])
    H_len, alphas_len = len(H), len(alphas)
    if H_len != alphas_len:
        return
    errors = []
    for i in range(iter):
        h, alpha = H[i], alphas[i]
        y_pred += alpha * h.predict(test_data)
        errors.append(error_rate(test_label, np.sign(y_pred)))
    y_pred = np.sign(y_pred)
    return y_pred.T, errors


# train adaboost
def adaboost_train(data, label, iter):
    data_len = data.shape[0]
    H = []
    alphas = []
    weights = np.ones(data_len) / data_len
    for i in range(iter):
        weak_classifier = DTC(max_depth=1, criterion='gini')
        weak_classifier.fit(data, label, sample_weight=weights.T)
        # tree.plot_tree(weak_classifier)
        # plt.show()
        label_pred = weak_classifier.predict(data)
        eps = weights.dot(label_pred != label) # epsilon
        alpha = (np.log(1 - eps) - np.log(eps)) / 2
        weights = weights * np.exp(- alpha * label * label_pred)
        weights = weights / weights.sum()
        H.append(weak_classifier)
        alphas.append(alpha)
    return H, alphas


# compute margin, weak_num is the number of weak classifier
def compute_margin(data, label, H, alphas, weak_num):
    margin_len = data.shape[0]
    margin = np.zeros(margin_len)
    sum_alphas = 0
    for i in range(weak_num):
        h, alpha = H[i], alphas[i]
        margin += alpha * h.predict(data)
        sum_alphas += alpha
    margin = margin * label / sum_alphas
    return margin


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = data_processing()
    H, alphas = adaboost_train(train_data, train_label, 100)
    _, train_error = adaboost_predict(train_data, train_label, H, alphas, 100)
    y_pred, test_error = adaboost_predict(test_data, test_label, H, alphas, 100)

    # draw error
    plt.plot(train_error, label='train_error')
    plt.plot(test_error, label='test_error')
    plt.xlabel('Hypothesis')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    # draw cumulative diagram
    weak_nums = [25, 50, 75, 100]
    for weak_num in weak_nums:
        margin = compute_margin(train_data, train_label, H, alphas, weak_num)
        values, base = np.histogram(margin, range=(-1, 1), bins=100)
        cumulative = np.cumsum(values)
        plt.step(base[:-1], cumulative, c='black', label='weak learner=%d'%(weak_num))
        plt.xlabel('margin')
        plt.ylabel('cumulative distribution')
        plt.grid()
        plt.legend()
        plt.show()
