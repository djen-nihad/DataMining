import numpy as np
import pandas as pd
from algorithms.Utile import Distance


def intra_cluster(X, labels):
    inertia = 0
    n_cluster = len(np.unique(labels, return_counts=False))
    if -1 in labels: n_cluster = n_cluster - 1
    for k in range(n_cluster):
        index_cluster_k = np.where(labels == k)
        cluster_k = X[index_cluster_k]
        center = np.mean(cluster_k, axis=0)
        for i in range(cluster_k.shape[0]):
            inertia += Distance.euclideanDistance(cluster_k[i, :], center)

    return inertia


def inter_cluster(X, labels, metric='euclidean', p=None):
    if metric == 'manhattan':
        distance = Distance.manhattanDistance
    elif metric == 'euclidean':
        distance = Distance.euclideanDistance
    elif metric == 'minkowski':
        distance = Distance.minkowskiDistance
    elif metric == 'cosine':
        distance = Distance.cosineDistance
    else:
        print('Metric not recognized, we will choose euclidean metric')
        distance = Distance.euclideanDistance

    n_cluster = len(np.unique(labels))
    if -1 in labels: n_cluster = n_cluster - 1

    centroids = []
    for i in range(n_cluster):
        centroids.append(np.mean(X[labels == i], axis=0))

    inertie = 0
    for i in range(n_cluster - 1):
        for j in range(i, n_cluster):
            inertie += distance(centroids[j], centroids[i])

    return inertie


def silhouette_score(X, labels, metric='euclidean', p=None):
    if metric == 'manhattan':
        distance = Distance.manhattanDistance
    elif metric == 'euclidean':
        distance = Distance.euclideanDistance
    elif metric == 'minkowski':
        distance = Distance.minkowskiDistance
    elif metric == 'cosine':
        distance = Distance.cosineDistance
    else:
        print('Metric not recognized, we will choose euclidean metric')
        distance = Distance.euclideanDistance

    n_clusters = len(np.unique(labels))
    if -1 in labels:
        n_clusters -= 1
    clusters = [X[np.where(labels == i)] for i in range(n_clusters)]

    silhouette_scores = []
    for i in range(X.shape[0]):
        cluster = labels[i]
        if cluster == -1:
            silhouette_scores.append(0)  # Silhouette score for noise points is 0
            continue

        indices_same_cluster = np.where((labels == cluster) & (np.arange(len(labels)) != i))[0]
        cluster_without_i = X[indices_same_cluster]

        # Calculate a_i
        if len(cluster_without_i) > 0:
            a_i = np.mean([distance(X[i, :], cluster_without_i[j, :], p) for j in range(cluster_without_i.shape[0])])
        else:
            a_i = 0

        # Calculate b_i
        b_i_values = [np.mean([distance(X[i, :], cluster[x, :], p) for x in range(cluster.shape[0])]) for k, cluster in
                      enumerate(clusters) if k != labels[i]]
        b_i = np.min(b_i_values) if b_i_values else 0

        # Calculate silhouette score for the current point
        if max(a_i, b_i) == 0:
            s_i = 0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)

        silhouette_scores.append(s_i)

    silhouette_avg = np.mean(silhouette_scores)

    return silhouette_avg


# EXACTITUDE
def accuracy_score(y_true, y_pred):
    return round(np.mean(y_true == y_pred), 4)


def precision_score(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    if (FP + TP) == 0: return 0
    else: return round(TP / (FP + TP), 4)


# Rappel
def recall_score(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    if (TP + FN) == 0: return 0
    else: return round(TP / (TP + FN), 4)


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if (precision + recall) == 0: return 0
    else: return round((2 * precision * recall) / (precision + recall), 4)


def specificity_score(y_true, y_pred):
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    if (TN + FP) == 0: return 0
    else: return round(TN / (TN + FP), 4)



def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(y_true.shape[0]):
        matrix[y_true[i], y_pred[i]] += 1

    return matrix


def np_to_pd_confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    n_classes = len(classes)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    print(confusion_matrix(y_true, y_pred))

    line = [classes[i] for i in range(n_classes)]
    columns = ["True Label X Predict Label"] + line
    matrix_pd = []
    for i in range(len(line)):
        new_line = [line[i]] + matrix[i]
        matrix_pd.append(new_line)
    return pd.DataFrame(matrix_pd, columns=columns)
