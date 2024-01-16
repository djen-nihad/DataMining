import numpy as np
import pandas as pd
from algorithms.Utile import Distance
from algorithms.metrics import *


class DBSCAN:
    def __init__(self, eps=1, min_samples=5, p=None, distance='euclidean'):
        # The maximum distance between two samples for one to be considered as in the
        # neighborhood of the other.
        self.eps = eps
        # The number of samples (or total weight) in a neighborhood for a point to be
        # considered as a core point.
        self.min_samples = min_samples
        if p is None:
            self.p = 2
        else:
            self.p = p
        if distance == 'manhattan':
            self.distance = Distance.manhattanDistance
        elif distance == 'euclidean':
            self.distance = Distance.euclideanDistance
        elif distance == 'minkowski':
            self.distance = Distance.minkowskiDistance
        else:
            self.distance = Distance.cosineDistance

    def find_neighbors(self):
        neighbors = [[i] for i in range(self.X.shape[0])]
        for i in range(self.X.shape[0] - 1):
            for j in range(i + 1, self.X.shape[0]):
                if self.distance == Distance.minkowskiDistance:
                    d = self.distance(self.X[i, :], self.X[j, :], self.p)
                else:
                    d = self.distance(self.X[i, :], self.X[j, :])
                if d < self.eps:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        return neighbors

    def fit(self, X):
        self.X = X
        self.n_cluster = -1

        # Initially, all samples are noise.
        self.labels_ = np.ones(X.shape[0], dtype=int) * -1

        neighborhoods = self.find_neighbors()

        for i in range(self.X.shape[0]):
            if self.labels_[i] != -1 or len(neighborhoods[i]) < self.min_samples:
                continue
            self.n_cluster += 1
            cluster = []
            self.labels_[i] = self.n_cluster
            cluster.extend(neighborhoods[i])
            while len(cluster) != 0:
                c = cluster.pop(0)
                if self.labels_[c] != -1: continue
                self.labels_[c] = self.n_cluster
                if len(neighborhoods[c]) >= self.min_samples:
                    cluster.extend(neighborhoods[c])
        self.n_cluster += 1

    def evaluate(self):
        columns = ["Silhouette Score", "Inertia ", "Inertie "]
        silhouette_avg = silhouette_score(self.X, self.labels_)
        interia = intra_cluster(self.X, self.labels_)
        inertie = inter_cluster(self.X, self.labels_)
        evaluate_list = [silhouette_avg, interia, inertie]
        return pd.DataFrame([evaluate_list], columns=columns)