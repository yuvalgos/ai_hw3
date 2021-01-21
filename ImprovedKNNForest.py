import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

from KNNForest import KNNForest


class ImprovedKNNForest(KNNForest):
    def predict(self, data_set, k_trees=None):
        if k_trees is None:
            k_trees = int(self.n_trees/4)
            if k_trees % 2 == 1:
                k_trees -= 1

        data_set_normalized = (data_set - self.fields_min) / \
                              (self.fields_max - self.fields_min)

        predictions = np.zeros([data_set.shape[0]])

        for i in range(len(predictions)):
            distances = np.linalg.norm(self.centroids_arr - data_set_normalized[i], axis=1)
            closest_trees_indices = np.argpartition(distances, k_trees)[:k_trees]
            closest_trees_indices = np.array(closest_trees_indices)

            distances = distances[closest_trees_indices]

            weights = np.ones([len(distances)])/np.square(distances)
            weights /= sum(weights)

            sample = np.array([data_set_normalized[i]])

            res = list(map(lambda x: x.predict(sample),
                           np.array(self.tree_list)[closest_trees_indices]
                           ))
            res = np.array(res)
            res[res == 0] = -1
            weighted_prediction = weights @ res  # dot product

            if weighted_prediction >= 1/2:
                predictions[i] = 1
            else:
                predictions[i] = 0

        return predictions


def experiment_m(m_params):
    """
    this function gets list : "m_params" of p to check
    and automaticly loads the dataset, performs 5-fold cross
    validation on each parameter and plots the result
    """
    kf = KFold(n_splits=5, random_state=205810179, shuffle=True)
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

    res = []
    for M in m_params:
        val_results = []
        for train_indices, val_indices in kf.split(train_set_data):
            train_fold_data = train_set_data[train_indices]
            val_fold_data = train_set_data[val_indices]
            train_fold_labels = train_set_labels[train_indices]
            val_fold_labels = train_set_labels[val_indices]

            sub_results = []
            for i in range(5):
                algo = KNNForest(M, 0.5)
                algo.fit(train_fold_data, train_fold_labels)

                predicted_diagnosis = algo.predict(val_fold_data)

                true_diagnosis_count = sum(predicted_diagnosis == val_fold_labels)
                sub_results.append(true_diagnosis_count/len(val_fold_labels))

            val_results.append(np.mean(sub_results))

        res.append(np.mean(val_results))

    plt.plot(m_params, res, 'o', m_params, res, 'k')
    plt.ylabel('accuracy')
    plt.xlabel('M')
    plt.title("p=0.5, K=M/4, (Avg of 5)")
    plt.show()

# experiment_m(range(5,100,5))

def experiment_k(k_params):
    """
    this function gets list : "k_params" of p to check
    and automaticly loads the dataset, performs 5-fold cross
    validation on each parameter and plots the result
    """
    kf = KFold(n_splits=5, random_state=205810179, shuffle=True)
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

    res = []
    for k in k_params:
        val_results = []
        for train_indices, val_indices in kf.split(train_set_data):
            train_fold_data = train_set_data[train_indices]
            val_fold_data = train_set_data[val_indices]
            train_fold_labels = train_set_labels[train_indices]
            val_fold_labels = train_set_labels[val_indices]

            sub_results = []
            for i in range(5):
                algo = KNNForest(75, 0.5)
                algo.fit(train_fold_data, train_fold_labels)

                predicted_diagnosis = algo.predict(val_fold_data, k_trees=k)

                true_diagnosis_count = sum(predicted_diagnosis == val_fold_labels)
                sub_results.append(true_diagnosis_count/len(val_fold_labels))

            val_results.append(np.mean(sub_results))

        res.append(np.mean(val_results))

    plt.plot(k_params, res, 'o', k_params, res, 'k')
    plt.ylabel('accuracy')
    plt.xlabel('K')
    plt.title("p=0.5, M=75, (Avg of 5)")
    plt.show()

# experiment_k(range(3,55,2))


def experiment_p(p_params):
    """
    this function gets list : "p_params" of p to check
    and automaticly loads the dataset, performs 5-fold cross
    validation on each parameter and plots the result
    """
    kf = KFold(n_splits=5, random_state=205810179, shuffle=True)
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

    res = []
    for p in p_params:
        val_results = []
        print("p=", p)
        for train_indices, val_indices in kf.split(train_set_data):
            train_fold_data = train_set_data[train_indices]
            val_fold_data = train_set_data[val_indices]
            train_fold_labels = train_set_labels[train_indices]
            val_fold_labels = train_set_labels[val_indices]

            sub_results = []
            for i in range(5):
                algo = KNNForest(50, p)
                algo.fit(train_fold_data, train_fold_labels)

                predicted_diagnosis = algo.predict(val_fold_data, k_trees=20)

                true_diagnosis_count = sum(predicted_diagnosis == val_fold_labels)
                sub_results.append(true_diagnosis_count/len(val_fold_labels))

            val_results.append(np.mean(sub_results))

        res.append(np.mean(val_results))

    plt.plot(p_params, res, 'o', p_params, res, 'k')
    plt.ylabel('accuracy')
    plt.xlabel('p')
    plt.title("K=53, M=75, (Avg of 5)")
    plt.show()

# experiment_p([0.3, 0.4, 0.5, 0.6, 0.7])

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

    train_set.drop_duplicates(features, inplace=True)

    train_set_data = train_set[features].to_numpy()
    train_set_labels = train_set['diagnosis'].to_numpy()
    test_set_data = test_set[features].to_numpy()
    test_set_labels = test_set['diagnosis'].to_numpy()

    # create 2nd order features :
    trans = PolynomialFeatures(degree=2, include_bias=False)
    train_set_data = trans.fit_transform(train_set_data)
    test_set_data = trans.fit_transform(test_set_data)

    # train Algorithm:
    algo = KNNForest(45, 0.5)
    algo.fit(train_set_data, train_set_labels)

    # get predictions on test set:
    predicted_diagnosis = algo.predict(test_set_data, 17)

    # calculate results:
    true_diagnosis_count = sum(predicted_diagnosis == test_set_labels)
    print(true_diagnosis_count / len(test_set_labels))


if __name__ == "__main__":
        main()
