import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# The train function, the l_r refers to the learning rate
def train(X_train, y_train, l_r):
    data_size = X_train.shape[0]
    model_weights = np.random.uniform(low=-5, high=5, size=(2, 1)) # Use uniform distribution to get random w
    model_intercept = np.random.uniform(low=-5, high=5, size=(1, 1))
    for i in range(500):
        w_T_x = X_train.dot(model_weights) + model_intercept # w^T * x
        y_t2m1 = 2 * y_train - 1 # 2y - 1
        exp_y_wx = np.exp(-y_t2m1 * w_T_x)
        dl_dw = np.sum(-y_t2m1 * X_train * exp_y_wx / (1 + exp_y_wx), axis=0) / data_size # delta loss/delta w
        dl_db = np.sum(-y_t2m1 * exp_y_wx / (1 + exp_y_wx), axis=0) / data_size
        dl_dw = dl_dw.reshape((2, 1))
        dl_db = dl_db.reshape((1, 1))
        model_weights -= l_r * dl_dw # gradient descent
        model_intercept -= l_r * dl_db
    return model_weights, model_intercept


# Sigmoid function
def sigmoid_t(x):
    return 1/(1 + np.exp(-x))


# predict function
def predict(X_valid, model_weights, model_intercept):
    y_predict_class = sigmoid_t(X_valid.dot(model_weights) + model_intercept)
    y_predict_class -= 0.5
    y_predict_class[y_predict_class < 0] = 0 # less than 1/2 is 0
    y_predict_class[y_predict_class > 0] = 1
    return y_predict_class


# separate blocks for the validation data
def get_next_train_valid(X_shuffled,  y_shuffled,  itr):
    val_id = itr # the number of the block
    block_size = int(X_shuffled.shape[0] / 5)
    X_valid = X_shuffled[block_size * val_id: block_size * (val_id + 1), :]
    y_valid = y_shuffled[block_size * val_id: block_size * (val_id + 1), :]
    X_train = np.vstack((X_shuffled[0: block_size * val_id, :], X_shuffled[block_size * (val_id + 1):, :] ))
    y_train = np.vstack((y_shuffled[0: block_size * val_id, :], y_shuffled[block_size * (val_id + 1):, :]))
    return X_train, y_train, X_valid, y_valid


# cross validation and training
def cross_validation_training(X_shuffled, y_shuffled, itr):
    Error_rates_val = []
    Error_rates_trn = []
    for i in range(itr):
        X_train, y_train, X_valid, y_valid = get_next_train_valid(X_shuffled,  y_shuffled,  i)
        l_r = (i + 1) * 0.01 # learning rate change with the iterations
        model_weights, model_intercept = train(X_train, y_train, l_r)
        y_pred_valid = predict(X_valid, model_weights, model_intercept)
        y_pred_train = predict(X_train, model_weights, model_intercept)
        confusion = confusion_matrix(y_valid, y_pred_valid)
        print("The confusion matrix for", i + 1, "validation set")
        print(confusion) # print confusion matrix of each iteration
        Errors = np.sum(np.abs(y_pred_valid - y_valid)) # 0 and 1 prediction, so sum up the predictions minus labels
        Err_rate = Errors / y_valid.shape[0]
        Error_rates_val.append(Err_rate)
        Errors = np.sum(np.abs(y_pred_train - y_train))
        Err_rate = Errors / y_train.shape[0]
        Error_rates_trn.append(Err_rate)
    return Error_rates_val, Error_rates_trn


def confusion_matrix(y_list, y_pred):
    label_classes_len = 2
    label_len = len(y_list)
    confusion = np.zeros((label_classes_len, label_classes_len))
    for i in range(label_len):
        matrix_row = int(y_list[i])
        matrix_col = int(y_pred[i])
        confusion[matrix_row][matrix_col] += 1
    return confusion



def read_data_rd(data_file, label_file):
    data_cord = np.genfromtxt(data_file, delimiter=',')
    data_label = np.genfromtxt(label_file)
    data_with_label = np.hstack((data_cord, data_label.reshape((-1, 1))))
    np.random.shuffle(data_with_label)
    X_shuffled = data_with_label[:, 0: 2]
    y_shuffled = data_with_label[:, -1].reshape((-1, 1))
    return X_shuffled, y_shuffled

if __name__ == '__main__':
    data_file = 'IRISFeat.csv'
    label_file = 'IRISlabel.csv'
    X_shuffled, y_shuffled = read_data_rd(data_file, label_file)
    Error_rates_val, Error_rates_trn = cross_validation_training(X_shuffled, y_shuffled, 5)
    line_trn = plt.plot(Error_rates_trn, label = 'Train error rate')
    line_val = plt.plot(Error_rates_val, label = 'Validation error rate')
    xint = [0, 1, 2, 3, 4]
    plt.xticks(xint)
    plt.xlabel('Cross validation')
    plt.ylabel('Error rates')
    plt.legend()
    plt.show()
