import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt

def gaussian_kernel(x1, x2, sigma=5):
    return np.exp(-np.linalg.norm(x1 - x2, axis=-1)**2 / (2 * (sigma ** 2)))

def rbf_svm_train(X, y, c, sigma):
    train_size = X.shape[0]
    K = np.zeros((train_size, train_size))
    for i in range(train_size):
        for j in range(train_size):
            K[i, j]= gaussian_kernel(X[i], X[j], sigma)
    P = cvxopt.matrix((y @ y.T) * K)
    q = cvxopt.matrix(-np.ones((train_size, 1)))
    G = cvxopt.matrix(np.vstack((-np.eye(train_size), np.eye(train_size))))
    h = cvxopt.matrix(np.hstack((np.zeros(train_size), np.ones(train_size) * c)))
    sol = cvxopt.solvers.qp(P, q, G, h)
    alpha = np.array(sol['x'])
    return alpha

def rbf_test1(X, sigma):
    train_size = X.shape[0]
    K = np.zeros((train_size, train_size))
    for i in range(train_size):
        for j in range(train_size):
            K[i, j]= gaussian_kernel(X[i], X[j], sigma)
    return K

def rbf_test2(X, sigma):
    train_size = X.shape[0]
    K = np.zeros((train_size, train_size))
    tmp = np.zeros((train_size, 2))
    for i in range(train_size):
        tmp[:] = X[i]
        K[i]= gaussian_kernel(X, tmp, sigma)
    return K

def predict(test_X, train_X, train_y, alpha, sigma):
    sv = (alpha > 1e-5).flatten()
    alpha_sv = alpha[sv]
    y_train_sv = train_y[sv].reshape(-1, 1)
    X_train_sv = train_X[sv]
    X_test_size = test_X.shape[0]
    X_sv_size = alpha_sv.shape[0]
    y_pred = np.zeros((X_test_size, 1))
    for i in range(X_test_size):
        tmp = 0
        for j in range(X_sv_size):
            tmp += alpha_sv[j] * y_train_sv[j] * gaussian_kernel(test_X[i], X_train_sv[j], sigma)
        y_pred[i]=np.sign(tmp)
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
    sigma = 5
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_next_train_valid(X_train_all, y_train_all, i)
        alpha = rbf_svm_train(X_train, y_train, c, sigma)
        y_pred_train = predict(X_train, X_train, y_train, alpha, sigma)
        y_pred_valid = predict(X_valid, X_train, y_train, alpha, sigma)
        y_pred_test = predict(X_test, X_train, y_train, alpha, sigma)
        train_accuracy_list.append(compute_accuracy(y_train, y_pred_train))
        cv_accuracy_list.append(compute_accuracy(y_valid, y_pred_valid))
        test_accuracy_list.append(compute_accuracy(y_test, y_pred_test))
    train_accuracy = np.mean(train_accuracy_list)
    cv_accuracy = np.mean(cv_accuracy_list)
    test_accuracy = np.mean(test_accuracy_list)
    return train_accuracy, cv_accuracy, test_accuracy

def get_next_train_valid(X_shuffled, y_shuffled, k, part_num = 8):
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
    X_train_all = data_label_train[:, 0: 2]
    y_train_all = data_label_train[:, -1].reshape((-1, 1))
    K1 = rbf_test2(X_train_all, 5)
    print("1")
    K2 =rbf_test2(X_train_all, 5)
    print("2")
    print(K1 == K2)
