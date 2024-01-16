from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from algorithms.UnsupervisedModel.Kmeans import KMeans
from algorithms.metrics import silhouette_score, inter_cluster


def experimenter_K(X, init):
    K = range(2, 11)
    silhouette_scores = []
    for k in K:
        model = KMeans(n_clusters=k, init=init)
        model.fit(X)
        labels = model.labels_
        silhouette_scores.append(silhouette_score(X, labels))

    plt.figure(figsize=(8, 6))
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score showing the optimal k')
    plt.legend()
    plt.show()


def experimenter_n_init(X):
    silhouette_scores = []
    n_inits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for n_init in n_inits:
        model = KMeans(n_clusters=2, init='k-means++', max_iter=10, n_init=n_init)
        score = 0
        nbriter = 3
        for i in range(nbriter):
            print(n_init, i)
            model.fit(X)
            labels = model.labels_
            score = score + silhouette_score(X, labels)
        score = score / nbriter
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 6))
    plt.plot(n_inits, silhouette_scores, 'bx-')  # Bleu
    plt.xlabel('Number of Initialization')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score showing the optimal Number of Initialization')
    plt.legend()
    plt.show()


def appliqueKmeans(X, n_clusters, n_init, init, max_iter, verbose=False):
    model = KMeans(n_clusters=n_clusters, n_init=n_init, init=init, max_iter=max_iter)
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    inertia = model.inertia_
    inertie = inter_cluster(X, labels)
    silhouette_avg = silhouette_score(X, labels)
    print('centroid : ', centroids)
    print(f'silhouette score = {silhouette_avg}')
    print(f'inertia  = {inertia}')
    print(f'inertie  = {inertie}')
    if verbose:
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(X)
        centroids_pca = pca.transform(centroids)
        # Visualize the clusters using the first two principal components
        plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', s=30, label='Data Points')
        # Add the centroid points in PCA space
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroids')
        plt.title(f'K-means Clustering (K={n_clusters})')

        plt.text(-3.5, 1.8, "Silhouette score: %.2f\nInertia: %f\nInertie: %.2f" % (silhouette_avg, inertia, inertie))
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.axis('equal')
        plt.show()


def KmeansWithDifferentValuesOfK(X):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(X)
    K = [2, 3, 4, 5, 6, 7, 8, 9]

    fig, axes = plt.subplots(3, len(K) + 1, figsize=(20, 10))

    axes[0, 0].scatter(data_pca[:, 0], data_pca[:, 1], s=30)
    axes[0, 0].set_title('Données Brutes')
    axes[0, 0].axis('equal')

    for i, eps in enumerate(K, 1):
        model = KMeans(n_clusters=eps)
        model.fit(X)
        labels = model.labels_

        axes[1, i].scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', s=30)
        axes[1, i].set_title(f'K={eps}')
        axes[1, i].axis('equal')

        # Validation Quantitative avec la silhouette
        silhouette_avg = silhouette_score(X, labels)
        print(f"Pour K={eps}, la silhouette moyenne est : {silhouette_avg}")

    plt.tight_layout()
    plt.show()