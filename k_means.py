import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, nr_classes = 3):
        self.nr_classes = nr_classes

    @staticmethod
    def distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def convergence_test(self, x1, x2):
        total_norm = 0
        for i in range(x1.shape[0]):
            c1 = x1[i, :]
            c2 = x2[i, :]
            total_norm += self.distance(c1, c2)
        if total_norm < 1e-3:
            return False
        return True

    def apply(self, X):
        dim = X.shape[1]
        colors = plt.cm.tab10.colors
        centroids = np.random.randn(self.nr_classes, dim)
        last_centroids = np.zeros((self.nr_classes, dim))
        iter = 0
        while self.convergence_test(centroids, last_centroids):
            plt.scatter(centroids[:, 0], centroids[:, 1], c=[colors[iter % 10]], marker="x")
            new_centroids = np.zeros((self.nr_classes, dim))
            nr_samples = np.zeros(self.nr_classes)
            for i in range(X.shape[0]):
                point = X[i, :]
                min_index = -1
                min_distance = 2000000000
                for j in range(self.nr_classes):
                    if min_distance > self.distance(point, centroids[j, :]):
                        min_index = j
                        min_distance = self.distance(point, centroids[j, :])
                new_centroids[min_index] += point
                nr_samples[min_index] += 1
            for i in range(self.nr_classes):
                if nr_samples[i] > 0:
                    new_centroids[i] /= nr_samples[i]
            last_centroids = centroids
            centroids = new_centroids
            iter += 1
        plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="x")
 
def test_kmeans():
    X, _ = make_blobs(n_samples=20, centers=3, cluster_std=0.6, random_state=42)

    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, c="blue")

    kmeans = KMeans(nr_classes=3)
    kmeans.apply(X)

    plt.show()

test_kmeans()