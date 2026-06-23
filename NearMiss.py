import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

class NearMissResampler:
    """Handles the undersampling algorithms."""
    def __init__(self, version=1, n_neighbors=3, m_neighbors=3):
        self.version = version
        self.n_neighbors = n_neighbors
        self.m_neighbors = m_neighbors

    def resample(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        
        minority_idx = np.where(y == minority_class)[0]
        majority_idx = np.where(y == majority_class)[0]
        
        selected_maj_idx = self._v3(X, minority_idx, majority_idx)
            
        final_idx = np.concatenate([minority_idx, selected_maj_idx])
        return X[final_idx], y[final_idx]

    def _v1(self, X, min_idx, maj_idx):
        print(f"Applying NearMiss-V1 (k={self.n_neighbors})...")
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(X[min_idx])
        distances, _ = nn.kneighbors(X[maj_idx])
        avg_distances = np.mean(distances, axis=1)
        return maj_idx[np.argsort(avg_distances)[:len(min_idx)]]

    def _v2(self, X, min_idx, maj_idx):
        print(f"Applying NearMiss-V2 (k={self.n_neighbors})...")
        dist_matrix = pairwise_distances(X[maj_idx], X[min_idx])
        k_farthest = np.partition(dist_matrix, -self.n_neighbors, axis=1)[:, -self.n_neighbors:]
        avg_distances = np.mean(k_farthest, axis=1)
        return maj_idx[np.argsort(avg_distances)[:len(min_idx)]]

    def _v3(self, X, min_idx, maj_idx):
        print(f"Applying NearMiss-V3 (k={self.n_neighbors}, m={self.m_neighbors})...")
        nn_maj = NearestNeighbors(n_neighbors=self.m_neighbors)
        nn_maj.fit(X[maj_idx])
        _, cand_sub_idx = nn_maj.kneighbors(X[min_idx])
        candidate_maj_idx = maj_idx[np.unique(cand_sub_idx.flatten())]
        
        n_to_select = len(min_idx)
        if len(candidate_maj_idx) <= n_to_select:
            return candidate_maj_idx
            
        nn_min = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn_min.fit(X[min_idx])
        distances, _ = nn_min.kneighbors(X[candidate_maj_idx])
        avg_distances = np.mean(distances, axis=1)
        return candidate_maj_idx[np.argsort(avg_distances)[:n_to_select]]
