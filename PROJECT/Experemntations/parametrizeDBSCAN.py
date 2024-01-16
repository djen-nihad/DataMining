from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from algorithms.UnsupervisedModel.DBSCAN import DBSCAN
from algorithms.metrics import *
import numpy as np


def appliqueDBSCAN(X, eps=0.3, min_samples=3, verbose=False):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    labels = model.labels_
    inertia = intra_cluster(X, labels)
    inertie = inter_cluster(X, labels)
    silhouette_avg = silhouette_score(X, labels)
    noise = np.count_nonzero(labels == -1)
    n_cluster = len(np.unique(labels))
    print(f'Number of clusters = {n_cluster}')
    print(f'Number of Noise = {noise}')
    print(f'silhouette score = {silhouette_avg}')
    print(f'inertia  = {inertia}')
    print(f'inertie  = {inertie}')
    if verbose:
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(X)
        # Visualize the clusters using the first two principal components
        # Create a custom colormap with 'black' included
        cmap = plt.cm.viridis
        cmap_list = [cmap(i) for i in range(cmap.N)]
        cmap_list[0] = (0, 0, 0, 1.0)  # Set the first color to black
        custom_cmap = ListedColormap(cmap_list)

        plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap=custom_cmap, s=30, label='Data Points')
        # Add the centroid points in PCA space
        plt.title(f'DBSCAN Clustering')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.axis('equal')
        plt.show()


def experimenter_min_samples(X, eps=0.38):
    min_s = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30]
    silhouette_scores = []
    for min in min_s:
        model = DBSCAN(eps=eps, min_samples=min)
        model.fit(X)
        labels = model.labels_
        silhouette_scores.append(silhouette_score(X, labels))
        print(min, silhouette_scores[-1])

    plt.figure(figsize=(8, 6))
    plt.plot(min_s, silhouette_scores, 'bx-')
    plt.xlabel('min_samples')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score showing the optimal min_samples')
    plt.legend()
    plt.show()


def experimenter_eps(X, min_samples=4):
    debut = 0.1
    fin = 0.4
    pas = 0.005

    Eps = np.arange(debut, fin + pas, pas)
    silhouette_scores = []
    for eps in Eps:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(X)
        labels = model.labels_
        silhouette_scores.append(silhouette_score(X, labels))
        print(eps, silhouette_scores[-1])

    plt.figure(figsize=(8, 6))
    plt.plot(Eps, silhouette_scores, 'bx-')
    plt.xlabel('eps')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score showing the optimal eps')
    plt.legend()
    plt.show()
