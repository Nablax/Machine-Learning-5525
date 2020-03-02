import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt

def gaussian_kernel(x1, x2, sigma=5):
    return np.exp(-np.linalg.norm(x1 - x2, axis=-1)**2 / (2 * (sigma ** 2)))

# draw heat map
def visualize_heat_map(c_list, sigma_list, accuracy, name):
    plt.title("{}".format(name))
    plt.imshow(1 - accuracy)
    ax, fig = plt.gca(), plt.gcf()
    plt.yticks(np.arange(len(c_list)), c_list)
    plt.xticks(np.arange(len(sigma_list)), sigma_list)
    plt.xlabel('sigma')
    plt.ylabel('C')
    plt.colorbar()
    ax.set_yticks(np.arange(len(c_list) + 1) - .5, minor=True)
    ax.set_xticks(np.arange(len(sigma_list) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.show()

# training and find alpha
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

# predict with alpha
def predict(test_X, train_X, train_y, alpha, sigma):
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

# compute accuracy
def compute_accuracy(y, y_pred):
    data_size = y.shape[0]
    if data_size != y_pred.shape[0]:
        return 0
    true_num = y[(y - y_pred)==0].shape[0]
    acc = true_num / data_size
    return acc

# after finding the best sigma and c, compute the validation error and test error here
def k_fold_best(traindata, testdata, k, c, sigma):
    X_test = testdata[:, 0: 2]
    y_test = testdata[:, -1].reshape((-1, 1))
    X_train_all = traindata[:, 0: 2]
    y_train_all = traindata[:, -1].reshape((-1, 1))
    cv_error_list, test_error_list = [], []
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_next_train_valid(X_train_all, y_train_all, i, k)
        alpha = rbf_svm_train(X_train, y_train, c, sigma)
        y_pred_valid = predict(X_valid, X_train, y_train, alpha, sigma)
        y_pred_test = predict(X_test, X_train, y_train, alpha, sigma)
        cv_error_list.append(1 - compute_accuracy(y_valid, y_pred_valid))
        test_error_list.append(1 - compute_accuracy(y_test, y_pred_test))
    return cv_error_list, test_error_list

# cross validation to find best sigma and c
def k_fold_cv(traindata, k, c, sigma):
    X_train_all = traindata[:, 0: 2]
    y_train_all = traindata[:, -1].reshape((-1, 1))
    cv_accuracy_list = []
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_next_train_valid(X_train_all, y_train_all, i, k)
        alpha = rbf_svm_train(X_train, y_train, c, sigma)
        y_pred_valid = predict(X_valid, X_train, y_train, alpha, sigma)
        cv_accuracy_list.append(compute_accuracy(y_valid, y_pred_valid))
    cv_accuracy = np.mean(cv_accuracy_list)
    return cv_accuracy

# get next validation set
def get_next_train_valid(X_shuffled, y_shuffled, k, part_num):
    val_id = k # the number of the block
    block_size = int(X_shuffled.shape[0] / part_num)
    X_valid = X_shuffled[block_size * val_id: block_size * (val_id + 1), :]
    y_valid = y_shuffled[block_size * val_id: block_size * (val_id + 1), :]
    X_train = np.vstack((X_shuffled[0: block_size * val_id, :], X_shuffled[block_size * (val_id + 1):, :] ))
    y_train = np.vstack((y_shuffled[0: block_size * val_id, :], y_shuffled[block_size * (val_id + 1):, :]))
    return X_train, y_train, X_valid, y_valid

# read data randomly
def read_data_rd(data_label_file):
    data_label = np.genfromtxt(data_label_file, delimiter=',')
    np.random.shuffle(data_label)
    return data_label

# split training data and test data
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
    sigma_list = [0.5, 5, 15, 50]
    c_len = len(c_list)
    s_len = len(sigma_list)
    k = 10
    train_accuracy_list, cv_accuracy_list, test_accuracy_list = \
        np.zeros((c_len, s_len)), np.zeros((c_len, s_len)), np.zeros((c_len, s_len))
    for i in range(c_len):
        for j in range(s_len):
            cv_accuracy = \
                k_fold_cv(data_label_train, k, c_list[i], sigma_list[j])
            cv_accuracy_list[i, j] = cv_accuracy
    c_list_label = ['0.0001','0.001','0.01','0.1','1','10','100','1000']
    sigma_list_label = ['0.5','5','15','50']
    visualize_heat_map(c_list_label, sigma_list_label, cv_accuracy_list, 'cv error heat map')
    print(cv_accuracy_list)
    c_best_at, s_best_at = np.unravel_index(np.argmax(cv_accuracy_list),cv_accuracy_list.shape)
    print(c_best_at, s_best_at)
    cv_error_list, test_error_list = \
        k_fold_best(data_label_train, data_label_test, k, c_list[c_best_at], sigma_list[s_best_at])

    plt.plot(test_error_list, label='test')
    plt.plot(cv_error_list, label='cv')
    plt.xlabel('folds')
    plt.ylabel('errors')
    plt.legend()
    plt.show()

