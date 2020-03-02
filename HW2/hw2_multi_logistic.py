import numpy as np
import matplotlib.pyplot as plt

def visualize_confusion_matrix(confusion, accuracy, label_classes, name):
    plt.title("{}, accuracy = {:.3f}".format(name, accuracy))
    plt.imshow(confusion)
    plt.colorbar()
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.show()

# get mini batches from the training data
def get_mini_batch(im_train, label_train, batch_size):
    # TO DO
    data_row = im_train.shape[0]
    data_col = im_train.shape[1]
    rd_size = int(data_col / batch_size)
    mini_batch_x = np.zeros((rd_size, data_row, batch_size))
    mini_batch_y = np.zeros((rd_size, 10, batch_size))
    rd_seed = np.arange(rd_size)
    # In this case rd_seed = range(24), mini batch will be 24 * 64 * 30, In the nth Iteration,
    # I shuffle the rd_seed and put im_train[24 * n + rd_seed[j]] into the nth element in every mini_batch
    for i in range(batch_size):
        np.random.shuffle(rd_seed)
        for j in range(rd_size):
            mini_batch_x[j, :, i] = im_train[:, i * rd_size + rd_seed[j]]
            mini_batch_y[j, label_train[:, i * rd_size + rd_seed[j]] - 1, i] = 1
    return mini_batch_x, mini_batch_y

def fc(x, w, b):
    # TO DO
    y = np.dot(w, x) + b
    return y

# cross entropy, loss = -y*ln(y_pred)
def loss_cross_entropy_softmax(x, y):
    # TO DO
    x_exp = np.exp(x)
    y_pred = x_exp / np.sum(x_exp)
    l = -np.sum(y * np.log(y_pred))
    dl_dy = y_pred - y
    return l, dl_dy


# full connect layer back propagation
def fc_backward(dl_dy, x, w, b, y):
    # TO DO
    dl_dx = np.dot(dl_dy, w)
    dl_dw = np.transpose(np.dot(x, dl_dy))
    dl_db = np.transpose(dl_dy)
    return dl_dx, dl_dw, dl_db


# train single layer perceptron
def minist_train(X, y, learning_rate = 1.5):
    # TO DO
    mini_batch_x = X
    mini_batch_y = y
    l_r = learning_rate
    d_r = 0.5
    w = np.random.normal(0, 1, (mini_batch_y.shape[1], mini_batch_x.shape[1]))
    b = np.zeros((mini_batch_y.shape[1], 1))
    mini_batch_num = mini_batch_x.shape[0]
    mini_batch_data_len = mini_batch_x.shape[1]
    mini_batch_label_len = mini_batch_y.shape[1]
    mini_batch_size = mini_batch_x.shape[2]
    k = 0
    for i in range(1, 5000):
        if i % 1000 == 0:
            l_r = l_r * d_r
        dl_dw = dl_db = 0
        for j in range(mini_batch_size):
            cur_batch_x = np.reshape(mini_batch_x[k, :, j], (mini_batch_data_len, 1))
            cur_batch_y = np.reshape(mini_batch_y[k, :, j], (mini_batch_label_len, 1))
            y_pred = fc(cur_batch_x, w, b)
            y_pred = np.transpose(y_pred)
            l, dl_dy = loss_cross_entropy_softmax(y_pred, np.transpose(cur_batch_y))
            dl_dx, dl_dw_tmp, dl_db_tmp = fc_backward(dl_dy, cur_batch_x, w, b, cur_batch_y)
            dl_dw += dl_dw_tmp
            dl_db += dl_db_tmp
        k += 1
        if k >= mini_batch_num:
            k = 0
        w = w - l_r * dl_dw / mini_batch_size
        b = b - l_r * dl_db / mini_batch_size
    return w, b

def read_data_mfeat(data_label_file):
    data = np.genfromtxt(data_label_file, delimiter=',', skip_header=1)
    X = (data[:, 1: -1]).T
    y = data[:, -1].reshape((1, -1)).astype(np.int64)
    return X, y

def minist_predict(w, b, X):
    X_len = X.shape[1]
    y_pred = np.zeros((X_len))
    for i in range(X_len):
        x = X[:, [i]]
        y = fc(x, w, b)
        y_pred[i] = np.argmax(y)
    return y_pred

if __name__ == '__main__':
    mnist_train_X, mnist_train_y = read_data_mfeat('mfeat_train.csv')
    mnist_test_X, mnist_test_y = read_data_mfeat('mfeat_test.csv')
    batch_size = 30
    mini_batch_x, mini_batch_y = get_mini_batch(mnist_train_X, mnist_train_y, batch_size)
    w, b = minist_train(mini_batch_x, mini_batch_y)
    np.savetxt('problem_6_weights', w)
    np.savetxt('problem_6_bias', b)
    y_pred = minist_predict(w, b, mnist_test_X)
    acc = 0
    confusion = np.zeros((10, 10))
    num_test = mnist_test_X.shape[1]
    for i in range(num_test):
        l_pred = int(y_pred[i])
        confusion[l_pred, mnist_test_y[0, i] - 1] = confusion[l_pred, mnist_test_y[0, i] - 1] + 1
        if l_pred == mnist_test_y[0, i] - 1:
            acc = acc + 1
    accuracy = acc / num_test
    for i in range(10):
        confusion[:, i] = confusion[:, i] / np.sum(confusion[:, i])

    label_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    visualize_confusion_matrix(confusion, accuracy, label_classes, 'Multi-class logistic regression')

