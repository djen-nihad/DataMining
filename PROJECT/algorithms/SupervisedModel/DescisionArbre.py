import numpy as np
import pandas as pd
from algorithms.metrics import accuracy_score, precision_score, recall_score, f1_score, specificity_score


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):

        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left  # left child
        self.right = right  # right child
        self.info_gain = info_gain

        # for leaf node
        self.value = value

class DecisionTree:
    def __init__(self, min_samples_split, max_depth):
        self.root = None   # the root

        # conditions to stop
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):  # recursive function

        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}  # dictionary to store the best split
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.left_right(dataset, feature_index, threshold)
                # check if children are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # return best split
        return best_split

    @staticmethod
    def left_right(dataset, feature_index, threshold):  # function to devise dataset

        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="entropy"):

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain

    @staticmethod
    def entropy(y):

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    @staticmethod
    def gini_index(y):

        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini

    @staticmethod
    def calculate_leaf_value(Y):

        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=" "):  # to print the tree

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % indent, end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % indent, end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):

        dataset = np.concatenate((X, np.array(Y).reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset)
    def fit2(self, X, Y):
        # function to train the tree

         # Check for duplicate indices in X
        unique_indices = np.unique(X, axis=0)
        num_samples, num_features = np.shape(X)

        if num_samples == unique_indices.shape[0]:  # No duplicate indices
            dataset = np.concatenate((X, np.array(Y).reshape(-1, 1)), axis=1)
        else:  # Duplicate indices
            # Create a new column for unique indices
            index_column = np.arange(num_samples).reshape(-1, 1)
            dataset = np.concatenate((X, index_column, np.array(Y).reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):  # to predict new dataset

        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):  # recursive function
        if tree.value is not None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
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
