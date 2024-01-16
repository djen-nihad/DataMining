import random
from algorithms.metrics import *
from algorithms.Utile import Distance


class KMeans:
    def __init__(self, n_clusters=8, p=None, distance='euclidean', max_iter=300, n_init=10,
                 init="k-means++", normalize=False):
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        if init == "k-means++":
            self.algorithm = self.initialize_centroids
        else:
            self.algorithm = self.generateCentroid
        if n_init == 'auto':
            if init == "k-means++":
                self.n_init = 1
            else:
                self.n_init = 10
        else:
            self.n_init = n_init

        if p is None:
            self.p = 2
        else:
            self.p = p
        self.normalize = normalize
        if distance == 'manhattan':
            self.distance = Distance.manhattanDistance
        elif distance == 'euclidean':
            self.distance = Distance.euclideanDistance
        elif distance == 'minkowski':
            self.distance = Distance.minkowskiDistance
        else:
            self.distance = Distance.cosineDistance

        self.inertia_ = None

    def generateCentroid(self):
        centroid_indices = random.sample(range(self.X.shape[0]), k=self.n_clusters)

        centroids = self.X[centroid_indices, :]

        return centroids

    def initialize_centroids(self):
        centroids = [self.X[np.random.choice(self.X.shape[0])]]

        for _ in range(1, self.n_clusters):
            distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in self.X])
            probabilities = distances / distances.sum()
            new_centroid = self.X[np.random.choice(self.X.shape[0], p=probabilities)]
            centroids.append(new_centroid)

        return centroids

    def fit(self, X):
        self.X = X

        for i in range(self.n_init):

            cluster_centers = self.algorithm()

            for k in range(self.max_iter):

                labels = self.assign_labels(cluster_centers)

                new_centroids = self.compute_centroids(labels)

                # if convergence
                if all(np.array_equal(old, new) for old, new in zip(cluster_centers, new_centroids)):
                    break

                cluster_centers = new_centroids

            inertia = intra_cluster(X, labels)
            self.nbr_iter = None

            if self.nbr_iter is None or self.nbr_iter < k:
                self.nbr_iter = k

            if self.inertia_ is None or self.inertia_ > inertia:

                self.cluster_centers_ = cluster_centers
                self.labels_ = labels
                self.inertia_ = inertia

    def assign_labels(self, cluster_centers):
        labels = []
        for i in range(self.X.shape[0]):
            if self.distance == Distance.minkowskiDistance:
                distanceI = [self.distance(self.X[i, :], c, self.p) for c in cluster_centers]
            else:
                distanceI = [self.distance(self.X[i, :], c) for c in cluster_centers]
            distanceI = np.array(distanceI)
            label = np.argmin(distanceI)
            labels.append(label)
        return np.array(labels)

    def compute_centroids(self, labels):
        centroids = []
        for k in range(self.n_clusters):
            index_centroid = np.where(labels == k)
            centroid = self.X[index_centroid, :]
            centroid = centroid[0]
            if len(centroid) == 0:
                centroid = self.X[np.random.choice(self.X.shape[0])]
            else: centroid = np.mean(centroid, axis=0)
            centroids.append(centroid)

        return centroids

    def evaluate(self):
        columns = ["Silhouette Score", "Inertia ", "Inertie "]
        silhouette_avg = silhouette_score(self.X, self.labels_)
        interia = self.inertia_
        inertie = inter_cluster(self.X, self.labels_)
        evaluate_list = [silhouette_avg, interia, inertie]
        return pd.DataFrame([evaluate_list], columns=columns)