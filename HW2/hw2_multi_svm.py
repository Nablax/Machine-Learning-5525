import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt

def gaussian_kernel(x1, x2, sigma=5):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * (sigma ** 2)))

def minist_svm_train(X, y, c, sigma):
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

def mnist_svm_predict(test_X, train_X, train_y, alphas, sigma):
    label_types_num = len(alphas)
    test_len = test_X.shape[0]
    y_pred_all = np.zeros((test_len, label_types_num))
    label_now = 0
    for alpha in alphas:
        sv = (alpha > 1e-5).flatten()
        alpha_sv = alpha[sv]
        y_train_sv = train_y[sv].reshape(-1, 1)
        X_train_sv = train_X[sv]
        X_test_size = test_X.shape[0]
        X_sv_size = alpha_sv.shape[0]
        y_pred = np.zeros(X_test_size)
        for i in range(X_test_size):
            tmp = 0
            for j in range(X_sv_size):
                tmp += alpha_sv[j] * y_train_sv[j] * gaussian_kernel(test_X[i], X_train_sv[j], sigma)
            y_pred[i] = tmp
        y_pred_all[:, label_now] = y_pred
        label_now += 1
    y_pred = np.argmax(y_pred_all, axis=1).reshape((-1, 1))
    y_pred += 1
    return y_pred

def compute_accuracy(y, y_pred):
    data_size = y.shape[0]
    if data_size != y_pred.shape[0]:
        return 0
    true_num = y[(y - y_pred)==0].shape[0]
    acc = true_num / data_size
    return acc

def read_data_mfeat(data_label_file):
    data = np.genfromtxt(data_label_file, delimiter=',', skip_header=1)
    X = data[:, 1: -1]
    y = data[:, -1].reshape((-1, 1)).astype(np.int)
    return X, y

def one_vs_all(X_train, y_train, X_test, y_test, C, label_types):
    data_len = y_train.shape[0]
    sigma = 5
    alphas = []
    for label in label_types:
        y_binary = np.ones((data_len, 1))
        y_binary[y_train != label] = -1
        alpha = minist_svm_train(X_train, y_binary, C, sigma)
        alphas.append(alpha)
    y_pred = mnist_svm_predict(X_test, X_train, y_train, alphas, sigma)
    acc = compute_accuracy(y_test, y_pred)
    return acc

if __name__ == "__main__":
    mnist_train_X, mnist_train_y = read_data_mfeat('mfeat_train.csv')
    mnist_test_X, mnist_test_y = read_data_mfeat('mfeat_test.csv')
    c_list = [10]
    train_accuracy_list, cv_accuracy_list, test_accuracy_list = [], [], []
    label_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for c in c_list:
        test_accuracy = \
            one_vs_all(mnist_train_X, mnist_train_y, mnist_test_X, mnist_test_y, c, label_types)
        test_accuracy_list.append(test_accuracy)
    print(test_accuracy_list)
    plt.plot(test_accuracy_list, label='test')
    plt.legend()
    plt.show()
