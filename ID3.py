import numpy as np
import pandas as pd
import math


def calc_entropy(p_Ci):
    if p_Ci == 0 or p_Ci == 1:
        return 0
    return -(p_Ci * math.log2(p_Ci) + (1-p_Ci) * math.log2(1-p_Ci))


def split(data_set, feature_index, threshold):
    true_indices = (data_set[:, feature_index] >= threshold)
    false_indices = (data_set[:, feature_index] < threshold)
    return true_indices, false_indices


class TreeLeaf:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, data):
        if self.prediction == 1:
            return np.ones(data.shape[0])
        else:
            return np.zeros(data.shape[0])


class TreeNode:  # which is not a leaf
    def __init__(self, depth):
        self.field = None
        self.threshold = None
        self.child_true = None
        self.child_false = None
        self.depth = depth

    def fit(self, data_set, labels):
        node_entropy = calc_entropy(np.count_nonzero(labels) / len(labels))
        min_ig = -42
        best_feature_index, best_threshold = None, None
        for i_feature in range(data_set.shape[1]):  # run on columns
            # prepare list of values to use as threshold:
            values = data_set[:, i_feature]
            values = np.unique(values)
            averages = (values[0:len(values)-1] + values[1:len(values)])/2
            for threshold in averages:
                true_indices, false_indices = split(data_set, i_feature, threshold)

                labels_true = labels[true_indices]
                labels_false = labels[false_indices]

                p_ci_true = (np.count_nonzero(labels_true) / len(labels_true))
                entropy_true = calc_entropy(p_ci_true)
                p_ci_false = (np.count_nonzero(labels_false) / len(labels_false))
                entropy_false = calc_entropy(p_ci_false)

                entropy = (entropy_false * len(labels_false) +
                           entropy_true * len(labels_true))/ len(labels)
                information_gain = node_entropy - entropy
                if information_gain >= min_ig:
                    min_ig = information_gain
                    best_feature_index, best_threshold = i_feature, threshold

        self.field = best_feature_index
        self.threshold = best_threshold

        # get the best split again:
        true_indices, false_indices = split(data_set, best_feature_index, best_threshold)
        data_true = data_set[true_indices, :]
        data_false = data_set[false_indices, :]
        labels_true = labels[true_indices]
        labels_false = labels[false_indices]

        true_data_ones = np.count_nonzero(labels_true)
        false_data_ones = np.count_nonzero(labels_false)

        # print(true_data_ones, len(labels_true), "--", false_data_ones, len(labels_false))

        # create children, if they are not leafs use fit to split them too:
        if true_data_ones == 0:
            self.child_true = TreeLeaf(0)
        elif true_data_ones == len(labels_true):
            self.child_true = TreeLeaf(1)
        else:  # not a leaf
            self.child_true = TreeNode(self.depth + 1)
            self.child_true.fit(data_true, labels_true)

        if false_data_ones == 0:
            self.child_false = TreeLeaf(0)
        elif false_data_ones == len(labels_false):
            self.child_false = TreeLeaf(1)
        else:  # not a leaf
            self.child_false = TreeNode(self.depth + 1)
            self.child_false.fit(data_false, labels_false)

    def predict(self, data_set):
        predictions = np.zeros(data_set.shape[0])

        true_indices, false_indices = split(data_set, self.field, self.threshold)
        predictions[true_indices] = self.child_true.predict(data_set[true_indices, :])
        predictions[false_indices] = self.child_false.predict(data_set[false_indices, :])

        return predictions


def main():
    # load and prepare both train and test set:
    train_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')

    train_set["diagnosis"].replace('B', 0, inplace=True)
    train_set["diagnosis"].replace('M', 1, inplace=True)
    test_set["diagnosis"].replace('B', 0, inplace=True)
    test_set["diagnosis"].replace('M', 1, inplace=True)

    features = train_set.columns.tolist()
    features.remove('diagnosis')

    train_set_data = train_set[features].to_numpy()
    train_set_labels = train_set['diagnosis'].to_numpy()
    test_set_data = test_set[features].to_numpy()
    test_set_labels = test_set['diagnosis'].to_numpy()

    # train tree:
    root = TreeNode(1)
    root.fit(train_set_data, train_set_labels)

    # get predictions on test set:
    predicted_diagnosis = root.predict(test_set_data)

    # calculate results:
    true_diagnosis_count = sum(predicted_diagnosis == test_set_labels)
    print(true_diagnosis_count / len(test_set_labels))



if __name__ == "__main__":
    main()
