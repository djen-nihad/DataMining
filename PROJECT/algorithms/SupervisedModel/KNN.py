from algorithms.Utile import Distance, normalization
from algorithms.metrics import accuracy_score, precision_score, recall_score, f1_score, specificity_score
from preprocessing import *
import numpy as np


class KNNClassifier(Distance):
    def __init__(self, n_neighbors, p=None, distance='euclidean'):
        self.X = None
        self.y = None
        self.n_neighbors = n_neighbors
        self.p = p
        if distance == 'manhattan':
            self.distance = Distance.manhattanDistance
        elif distance == 'euclidean':
            self.distance = Distance.euclideanDistance
        elif distance == 'minkowski':
            self.distance = Distance.minkowskiDistance
        else:
            self.distance = Distance.cosineDistance

    def fit(self, X, y):
       self.X = X
       self.y = y

    def predict(self, X_test):
        try:
            y_pred = []
            for instance in X_test:
                distances = [self.distance(instance, x, self.p) for x in self.X]
                k_indices = np.argsort(distances)[:self.n_neighbors]
                k_labels = self.y[k_indices]
                labels_unique = np.unique(k_labels, return_counts=True)
                y = np.argmax(labels_unique[1])
                y_pred.append(labels_unique[0][y])
            return np.array(y_pred)
        except:
            print("ERROR IN FIT FUNCTIONS. CALL FIT FUNCTION")

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
