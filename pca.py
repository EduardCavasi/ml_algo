import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
    def get_covariance_matrix(self, X):
        nr_samples = X.shape[0]
        return 1 / (nr_samples - 1) * (X.T @ X)
    def get_new_basis(self, cov_matrix):
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        D = eigenvectors[:, -1:-self.n_components-1:-1]
        return D
    def fit_transform(self, X):
        cov_matrix = self.get_covariance_matrix(X)
        D = self.get_new_basis(cov_matrix)
        X_pca = X @ D
        return X_pca

def test_pca():
    np.random.seed(42)
    X = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T

    plt.figure(figsize = (6, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha =0.5, c='blue')

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, c='red')

    validation_pca = sk.decomposition.PCA(n_components=2)
    X_pca_validation = validation_pca.fit_transform(X)

    plt.scatter(X_pca_validation[:, 0], X_pca_validation[:, 1], alpha=0.5, c='green')
    plt.show()

test_pca()