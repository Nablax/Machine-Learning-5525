import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt

def gaussian_kernel(x1, x2, sigma=5):
    return np.exp(-np.linalg.norm(x1 - x2, axis=-1)**2 / (2 * (sigma ** 2)))

def rbf_svm_train(X, y, c, sigma):
    train_size = X.shape[0]
    K = np.zeros((train_size, train_size))
    tmp_X = np.zeros((train_size, 2))
    for i in range(train_size):
        tmp_X[:] = X[i]
        K[i]= gaussian_kernel(X, tmp_X, sigma)
    P = cvxopt.matrix((y @ y.T) * K)
    q = cvxopt.matrix(-np.ones((train_size, 1)))
    G = cvxopt.matrix(np.vstack((-np.eye(train_size), np.eye(train_size))))
    h = cvxopt.matrix(np.hstack((np.zeros(train_size), np.ones(train_size) * c)))
    sol = cvxopt.solvers.qp(P, q, G, h)
    alpha = np.array(sol['x'])
    return alpha

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

def predict2(test_X, train_X, train_y, alpha, sigma):
    sv = (alpha > 1e-5).flatten()
    alpha_sv = alpha[sv]
    y_train_sv = train_y[sv].reshape(-1, 1)
    X_train_sv = train_X[sv]
    X_test_size = test_X.shape[0]
    X_sv_size = alpha_sv.shape[0]
    y_pred = np.zeros((X_test_size, 1))
    tmp = np.zeros_like(X_train_sv)
    for i in range(X_test_size):
        tmp[:] = test_X[i]
        y_pred[i] = np.sum(alpha_sv * y_train_sv * gaussian_kernel(tmp, X_train_sv, sigma).reshape((-1, 1)))
    y_pred = np.sign(y_pred)
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
        y_pred_train = predict2(X_train, X_train, y_train, alpha, sigma)
        y_pred_valid = predict2(X_valid, X_train, y_train, alpha, sigma)
        y_pred_test = predict2(X_test, X_train, y_train, alpha, sigma)
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
    c_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    train_accuracy_list, cv_accuracy_list, test_accuracy_list = [], [], []
    for c in c_list:
        train_accuracy, cv_accuracy, test_accuracy = k_fold_cv(data_label_train, data_label_test, 8, c)
        train_accuracy_list.append(train_accuracy)
        cv_accuracy_list.append(cv_accuracy)
        test_accuracy_list.append(test_accuracy)
    print(train_accuracy_list)
    print(test_accuracy_list)
    print(cv_accuracy_list)
    plt.plot(train_accuracy_list, label='train')
    plt.plot(test_accuracy_list, label='test')
    plt.plot(cv_accuracy_list, label='cv')
    plt.legend()
    plt.show()
