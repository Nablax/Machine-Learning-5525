import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt

def svmfit(X, y, c):
    train_size = X.shape[0]
    y_m_X_train = y * X
    P = cvxopt.matrix(y_m_X_train.dot(y_m_X_train.T))
    q = cvxopt.matrix(-np.ones((train_size, 1)))
    G = cvxopt.matrix(np.vstack((-np.eye(train_size),np.eye(train_size))))
    h = cvxopt.matrix(np.hstack((np.zeros(train_size), np.ones(train_size) * c)))
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h)
    lambda_sol = np.array(sol['x'])
    w = ((y * lambda_sol).T @ X).reshape(-1, 1)
    return w


def predict(X, w):
    y_pred = np.sign(X @ w)
    return y_pred

def compute_accuracy(y, y_pred):
    data_size = y.shape[0]
    if data_size != y_pred.shape[0]:
        return 0
    true_num = y[(y - y_pred)==0].shape[0]
    acc = true_num / data_size
    return acc


def k_fold_cv(traindata, testdata, k, c):
    X_test = testdata[:, 0: 2]
    y_test = testdata[:, -1].reshape((-1, 1))
    X_train_all = traindata[:, 0: 2]
    y_train_all = traindata[:, -1].reshape((-1, 1))
    train_accuracy_list, cv_accuracy_list, test_accuracy_list = [], [], []
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_next_train_valid(X_train_all, y_train_all, i, k)
        w = svmfit(X_train, y_train, c)
        y_pred_train = predict(X_train, w)
        y_pred_valid = predict(X_valid, w)
        y_pred_test = predict(X_test, w)
        train_accuracy_list.append(compute_accuracy(y_train, y_pred_train))
        cv_accuracy_list.append(compute_accuracy(y_valid, y_pred_valid))
        test_accuracy_list.append(compute_accuracy(y_test, y_pred_test))
    train_accuracy = np.mean(train_accuracy_list)
    cv_accuracy = np.mean(cv_accuracy_list)
    test_accuracy = np.mean(test_accuracy_list)
    return train_accuracy, cv_accuracy, test_accuracy

def get_next_train_valid(X_shuffled, y_shuffled, k, part_num = 10):
    val_id = k # the number of the block
    block_size = int(X_shuffled.shape[0] / part_num)
    X_valid = X_shuffled[block_size * val_id: block_size * (val_id + 1), :]
    y_valid = y_shuffled[block_size * val_id: block_size * (val_id + 1), :]
    X_train = np.vstack((X_shuffled[0: block_size * val_id, :], X_shuffled[block_size * (val_id + 1):, :] ))
    y_train = np.vstack((y_shuffled[0: block_size * val_id, :], y_shuffled[block_size * (val_id + 1):, :]))
    return X_train, y_train, X_valid, y_valid

def read_data_rd(data_label_file):
    data_label = np.genfromtxt(data_label_file, delimiter=',')
    np.random.shuffle(data_label)
    return data_label

def split_data_rd(data_label, test_percent):
    if test_percent < 0 or test_percent > 0.5:
        test_percent = 0.2
    data_size = data_label.shape[0]
    test_nums = int(data_size * test_percent)
    data_label_train = data_label[test_nums: data_size, :]
    data_label_test = data_label[0: test_nums, :]
    return data_label_train, data_label_test

if __name__ == "__main__":
    data_label_file = 'hw2data.csv'
    data_label = read_data_rd(data_label_file)
    data_label_train, data_label_test = split_data_rd(data_label, 0.2)
    c_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    train_accuracy_list, cv_accuracy_list, test_accuracy_list = [], [], []
    for c in c_list:
        train_accuracy, cv_accuracy, test_accuracy = k_fold_cv(data_label_train, data_label_test, 8, c)
        train_accuracy_list.append(train_accuracy)
        cv_accuracy_list.append(cv_accuracy)
        test_accuracy_list.append(test_accuracy)
    c_list_label = ['0.0001', '0.001', '0.01', '0.1', '1', '10', '100', '1000']
    plt.xticks(np.arange(len(c_list_label)), c_list_label)
    plt.plot(train_accuracy_list, label='train')
    plt.plot(test_accuracy_list, label='test')
    plt.plot(cv_accuracy_list, label='cv')
    plt.legend()
    plt.show()
