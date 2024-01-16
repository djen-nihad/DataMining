from algorithms.SupervisedModel.DescisionArbre import DecisionTree
from scipy.stats import mode
import numpy as np
import pandas as pd
from algorithms.metrics import accuracy_score, precision_score, recall_score, f1_score, specificity_score


class RandomForest:
    def __init__(self, max_tree, min_samples_split, max_depth):
        self.max_tree = max_tree
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def bootstrapping(self, dataset):
        np.random.seed(44)  # makes randomness reproducible
        bootstrapped_data = []
        nb_samples, nb_features = dataset.shape
        selected_features = int(np.sqrt(nb_features))  # number of selected features for each subset
        for i in range(self.max_tree):
            subset_indices = np.random.choice(nb_samples, nb_samples, replace=True)
            subset_columns = np.random.choice(nb_features, size=selected_features, replace=False)
            subset = dataset.iloc[subset_indices, subset_columns]
            bootstrapped_data.append(subset)
        return bootstrapped_data

    def make_forest(self, data_train, y_training):
        subsets = self.bootstrapping(data_train)
        forest = []
        for subset in subsets:
            TREE = DecisionTree(self.min_samples_split, self.max_depth)
            TREE.fit2(subset, y_training)
            forest.append(TREE)
        return forest

    @staticmethod
    def prediction(forest, new_data):
        predictions = []
        for tree in forest:
            tree_prediction = tree.predict(np.array(new_data))
            predictions.append(tree_prediction)

        final_predictions, _ = mode(predictions, axis=0)
        final_predictions = final_predictions.flatten()
        return final_predictions

    def evaluate(self, forest, X_test, y_test):
        predictions = RandomForest.prediction(forest, X_test)
        columns = ["Class", "Accuracy", "Precision", "Recall", "F1-score", "Specificity"]
        evaluate_list = []
        classes = np.unique(y_test)
        for classe in classes:
            y_test_classe = (y_test == classe).astype(int)
            predictions_classe = (predictions == classe).astype(int)
            accuracy = accuracy_score(y_test_classe, predictions_classe)
            accuracy = round(accuracy * 100, 2)
            precision = precision_score(y_test_classe, predictions_classe)
            precision = round(precision * 100, 2)
            recall = recall_score(y_test_classe, predictions_classe)
            recall = round(recall * 100, 2)
            f1 = f1_score(y_test_classe, predictions_classe)
            f1 = round(f1 * 100, 2)
            specificity = specificity_score(y_test_classe, predictions_classe)
            specificity = round(specificity * 100, 2)
            evaluate_list.append([classe, accuracy, precision, recall, f1, specificity])
        accuracy = accuracy_score(y_test, predictions)
        accuracy = round(accuracy * 100, 2)
        precision = precision_score(y_test, predictions)
        precision = round(precision * 100, 2)
        recall = recall_score(y_test, predictions)
        recall = round(recall * 100, 2)
        f1 = f1_score(y_test, predictions)
        f1 = round(f1 * 100, 2)
        specificity = specificity_score(y_test, predictions)
        specificity = round(specificity * 100, 2)
        evaluate_list.append(["Global", accuracy, precision, recall, f1, specificity])

        return pd.DataFrame(evaluate_list, columns=columns), predictions
