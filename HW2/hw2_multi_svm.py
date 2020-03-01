import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt

def visualize_confusion_matrix(confusion, accuracy, label_classes, name):
    plt.title("{}, accuracy = {:.3f}".format(name, accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.show()

def gaussian_kernel(x1, x2, sigma=5):
    return np.exp(-np.linalg.norm(x1 - x2, axis=-1)**2 / (2 * (sigma ** 2)))


def minist_svm_train(X, y, c, sigma):
    train_size = X.shape[0]
    K = np.zeros((train_size, train_size))
    tmp = np.zeros_like(X)
    for i in range(train_size):
        tmp[:] = X[i]
        K[i]= gaussian_kernel(X, tmp, sigma)
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

        tmp = np.zeros_like(X_train_sv)
        for i in range(X_test_size):
            tmp[:] = test_X[i]
            y_pred[i] = np.sum(alpha_sv * y_train_sv * gaussian_kernel(tmp, X_train_sv, sigma).reshape((-1, 1)))
        y_pred_all[:, label_now] = y_pred
        label_now += 1
    y_pred = np.argmax(y_pred_all, axis=1).reshape((-1, 1))
    return y_pred

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
    return y_pred

if __name__ == "__main__":
    mnist_train_X, mnist_train_y = read_data_mfeat('mfeat_train.csv')
    mnist_test_X, mnist_test_y = read_data_mfeat('mfeat_test.csv')
    c_list = [10]
    train_accuracy_list, cv_accuracy_list, test_accuracy_list = [], [], []
    label_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for c in c_list:
        y_pred = \
            one_vs_all(mnist_train_X, mnist_train_y, mnist_test_X, mnist_test_y, c, label_types)
        acc = 0
        confusion = np.zeros((10, 10))
        num_test = mnist_test_X.shape[1]
        for i in range(num_test):
            confusion[y_pred[i], mnist_test_y[i, 0] - 1] += 1

            if y_pred[i] == mnist_test_y[i, 0] - 1:
                acc = acc + 1
        accuracy = acc / num_test
        for i in range(10):
            confusion[:, i] = confusion[:, i] / np.sum(confusion[:, i])
        label_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        visualize_confusion_matrix(confusion, accuracy, label_classes, 'Single-layer Perceptron Confusion Matrix')

