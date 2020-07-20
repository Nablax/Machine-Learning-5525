from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# process data, transfer {0, 1} label to {-1, 1}
def data_processing():
    train_data_all = np.genfromtxt('health_train.csv', delimiter=',')[1:, :]
    test_data_all = np.genfromtxt('health_test.csv', delimiter=',')[1:, :]
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


# predict random forest
def random_forest_predict(Trees, features, test_data):
    y_pred = np.zeros(test_data.shape[0])
    for tree, feature in zip(Trees, features):
        y_pred += tree.predict(test_data[:, feature])
    return np.sign(y_pred).T


# train random forest
def random_forest_train(data, label, feature_num, max_tree):
    data_len = data.shape[0]
    feature_len = data.shape[1]
    Trees = []
    features = []
    rd_prop = 0.25
    for i in range(max_tree):
        # random samples
        rd_samples = np.random.choice(data_len, int(rd_prop * data_len), replace=False)
        # random features
        rd_features = np.random.choice(feature_len, feature_num, replace=False)
        samples_label = label[rd_samples]
        samples_data = data[rd_samples, :]
        samples_data = samples_data[:, rd_features]
        features.append(rd_features)
        weak_classifier = DTC(criterion='gini')
        weak_classifier.fit(samples_data, samples_label)
        Trees.append(weak_classifier)
    return Trees, features

if __name__ == '__main__':
    train_data, train_label, test_data, test_label = data_processing()
    # different number of features
    feature_choice = [50, 100, 150, 200, 250]
    train_acc = []
    test_acc = []
    for feature in feature_choice:
        Trees, features = random_forest_train(train_data, train_label, feature, 100)
        y_pred = random_forest_predict(Trees, features, train_data)
        train_acc.append(1 - error_rate(train_label, y_pred))
        y_pred = random_forest_predict(Trees, features, test_data)
        test_acc.append(1 - error_rate(test_label, y_pred))

    plt.plot(train_acc, label='train accuracy')
    plt.plot(test_acc, label='test accuracy')
    plt.xlabel('Feature num')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(5), ['50', '100', '150', '200', '250'])
    plt.legend()
    plt.show()

    # different number of trees
    train_acc = []
    test_acc = []
    tree_set = [10, 20, 40, 80, 100]
    for tree_size in tree_set:
        Trees, features = random_forest_train(train_data, train_label, 250, tree_size)
        y_pred = random_forest_predict(Trees, features, train_data)
        train_acc.append(1 - error_rate(train_label, y_pred))
        y_pred = random_forest_predict(Trees, features, test_data)
        test_acc.append(1 - error_rate(test_label, y_pred))

    plt.plot(train_acc, label='train accuracy')
    plt.plot(test_acc, label='test accuracy')
    plt.xlabel('Tree num')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(5), ['10', '20', '40', '80', '100'])
    plt.legend()
    plt.show()
