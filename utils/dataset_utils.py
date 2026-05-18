import os.path
from torch.utils.data  import Dataset
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle
import pandas as pd


class WSIGraphDataset(Dataset):

    def __init__(self, patch_embed, patch_label, patch_majorid, patch_subid, patch_coords, min_patches=1000, n_clusters=100):

        self.patch_embed = patch_embed
        self.patch_label = patch_label
        self.patch_majorid = patch_majorid
        self.patch_subid = patch_subid
        self.patch_coords = patch_coords
        self.min_patches = min_patches
        self.n_clusters = n_clusters
        self.all_pairs = np.hstack((patch_majorid.reshape(-1, 1), patch_subid.reshape(-1, 1)))
        self.unique_pairs = np.unique(self.all_pairs, axis=0)

        print("Filtering slides based on minimum patch count...")
        self.valid_unique_pairs = []
        print("exclude slide id:")
        for pair in self.unique_pairs:

            flags = np.isin(self.all_pairs, pair).all(axis=1)
            labels = self.patch_label[flags][0]
            num_patches = np.sum(flags)
            if num_patches >= self.min_patches:
                self.valid_unique_pairs.append(pair)

            else:
                print(f"{pair[0]},{pair[1]},{labels}")
                continue
        self.valid_unique_pairs = np.array(self.valid_unique_pairs)
        print(f"Filtering complete. Kept {len(self.valid_unique_pairs)} out of {len(self.unique_pairs)} slides.")

    def __len__(self):
        return self.valid_unique_pairs.shape[0]

    def __getitem__(self, idx):
        selected_slide = self.valid_unique_pairs[idx]
        flags = np.isin(self.all_pairs, selected_slide).all(axis=1)
        feats = self.patch_embed[flags]
        coords = self.patch_coords[flags]
        labels = self.patch_label[flags]

        pyg_data, cluster_patch_indices = self._create_pyg_data(feats, coords, idx,selected_slide)

        return pyg_data, torch.tensor(labels[0],dtype=torch.long), selected_slide, cluster_patch_indices, coords, feats

    def _create_pyg_data(self, features, positions, batch_idx,selected_slide):

        cluster_save_root = os.environ.get('PITNET_KMEANS_CACHE', './cache/kmeans')
        os.makedirs(cluster_save_root, exist_ok=True)

        cluster_save_pth = os.path.join(cluster_save_root, f'{str(selected_slide[0])}_{str(selected_slide[1])}.pkl')

        if not os.path.exists(cluster_save_pth):
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10).fit(features)
            print(f"saving {cluster_save_pth}")
            with open(cluster_save_pth, 'wb') as f:
                pickle.dump(kmeans, f)
        else:
            with open(cluster_save_pth, 'rb') as f:
                kmeans = pickle.load(f)

        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
        representative_features = []
        representative_positions = []
        cluster_patch_indices = {i: [] for i in range(self.n_clusters)}
        for i in range(self.n_clusters):
            indices_in_cluster = np.where(cluster_labels == i)[0]
            if len(indices_in_cluster) == 0:
                continue
            cluster_patch_indices[i].extend(indices_in_cluster.tolist())
            features_in_cluster = features[indices_in_cluster]
            cluster_center = cluster_centers[i].reshape(1, -1)
            distances = cdist(features_in_cluster, cluster_center)
            closest_point_index_in_cluster = np.argmin(distances)
            absolute_index = indices_in_cluster[closest_point_index_in_cluster]
            representative_features.append(features[absolute_index])
            representative_positions.append(positions[absolute_index])

        features_tensor = torch.tensor(np.array(representative_features), dtype=torch.float)
        positions_tensor = torch.tensor(np.array(representative_positions), dtype=torch.float)

        edge_index = knn_graph(positions_tensor, k=8, loop=True, batch=None) #1026

        data = Data(
            x=features_tensor,
            pos=positions_tensor,
            edge_index=edge_index,
            batch=torch.full((features_tensor.shape[0],), batch_idx, dtype=torch.long)
        )

        return data, cluster_patch_indices

