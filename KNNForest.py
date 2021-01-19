from ID3 import TreeNode
import numpy as np
import pandas as pd


class KNNForest:
    def __init__(self, n_trees, p):
        self.n_trees = n_trees
        self.p = p

        self.tree_list = [TreeNode() for _ in range(0, n_trees)]
        self.centroids_arr = None
        self.fields_max = None
        self.fields_min = None

    def fit(self, data_set, labels):
        indices = np.arange(len(labels))
        subsets_size = int(self.p * len(labels))
        self.centroids_arr = np.zeros([self.n_trees, data_set.shape[1]])
        self.fields_max = np.max(data_set, axis=0)
        self.fields_min = np.min(data_set, axis=0)

        data_set_normalized = (data_set - self.fields_min) / \
                              (self.fields_max - self.fields_min)

        for i in range(0, self.n_trees):
            # choose random data :
            np.random.shuffle(indices)
            subset_data = data_set_normalized[indices[:subsets_size], :]
            subset_labels = labels[indices[:subsets_size]]

            # train the tree:
            self.tree_list[i].fit(subset_data, subset_labels)

            # calculate the centroid:
            self.centroids_arr[i] = np.mean(subset_data, axis=0)

    def predict(self, data_set, k_trees=None):
        if k_trees is None:
            k_trees = self.n_trees/2

        data_set_normalized = (data_set - self.fields_min) / \
                              (self.fields_max - self.fields_min)

        predictions = np.zeros([data_set.shape[0]])

        for i in range(len(predictions)):
            distances = np.linalg.norm(self.centroids_arr - data_set_normalized[i], axis=1)
            closest_trees_indices = np.argpartition(distances, k_trees)[:k_trees]
            closest_trees_indices = np.array(closest_trees_indices)
            sample = np.array([data_set_normalized[i]])

            res = list(map(lambda x: x.predict(sample),
                           np.array(self.tree_list)[closest_trees_indices]
                           ))

            if sum(res) > k_trees/2:
                predictions[i] = 1
            else:
                predictions[i] = 0

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

    # train Algorithm:
    algo = KNNForest(100, 0.666)
    algo.fit(train_set_data, train_set_labels)

    # get predictions on test set:
    predicted_diagnosis = algo.predict(test_set_data, 20)

    # calculate results:
    true_diagnosis_count = sum(predicted_diagnosis == test_set_labels)
    print(true_diagnosis_count / len(test_set_labels))


if __name__ == "__main__":
    main()
