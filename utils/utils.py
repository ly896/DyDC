import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


def get_neighbor_finder(data, uniform=False, max_node_idx=None):
    if max_node_idx is None:
        max_node_idx = max(data.sources.max(), data.destinations.max())

    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(
        data.sources, data.destinations, data.edge_idxs, data.timestamps
    ):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []

        for neighbors in adj_list:
            sorted_neighbors = sorted(neighbors, key=lambda item: item[2])
            self.node_to_neighbors.append(np.array([item[0] for item in sorted_neighbors]))
            self.node_to_edge_idxs.append(np.array([item[1] for item in sorted_neighbors]))
            self.node_to_edge_timestamps.append(
                np.array([item[2] for item in sorted_neighbors])
            )

        self.uniform = uniform
        self.seed = seed
        self.random_state = np.random.RandomState(seed) if seed is not None else None

    def find_before(self, src_idx, cut_time):
        index = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        return (
            self.node_to_neighbors[src_idx][:index],
            self.node_to_edge_idxs[src_idx][:index],
            self.node_to_edge_timestamps[src_idx][:index],
        )

    def get_temporal_neighbor_tqdm(self, source_nodes, timestamps, n_neighbors=20):
        iterator = tqdm(enumerate(zip(source_nodes, timestamps)), total=len(source_nodes))
        return self._get_temporal_neighbor(source_nodes, timestamps, n_neighbors, iterator)

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        iterator = enumerate(zip(source_nodes, timestamps))
        return self._get_temporal_neighbor(source_nodes, timestamps, n_neighbors, iterator)

    def _get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors, iterator):
        assert len(source_nodes) == len(timestamps)
        effective_neighbors = n_neighbors if n_neighbors > 0 else 1
        neighbors = np.zeros((len(source_nodes), effective_neighbors), dtype=np.int32)
        edge_times = np.zeros((len(source_nodes), effective_neighbors), dtype=np.float32)
        edge_idxs = np.zeros((len(source_nodes), effective_neighbors), dtype=np.int32)

        for row, (source_node, timestamp) in iterator:
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(
                source_node, timestamp
            )

            if len(source_neighbors) == 0 or n_neighbors <= 0:
                continue

            if self.uniform:
                sampled_idx = self._sample_indices(len(source_neighbors), n_neighbors)
                selected_neighbors = source_neighbors[sampled_idx]
                selected_edge_times = source_edge_times[sampled_idx]
                selected_edge_idxs = source_edge_idxs[sampled_idx]
                order = selected_edge_times.argsort()
                selected_neighbors = selected_neighbors[order]
                selected_edge_times = selected_edge_times[order]
                selected_edge_idxs = selected_edge_idxs[order]
            else:
                selected_neighbors = source_neighbors[-n_neighbors:]
                selected_edge_times = source_edge_times[-n_neighbors:]
                selected_edge_idxs = source_edge_idxs[-n_neighbors:]

            offset = n_neighbors - len(selected_neighbors)
            neighbors[row, offset:] = selected_neighbors
            edge_times[row, offset:] = selected_edge_times
            edge_idxs[row, offset:] = selected_edge_idxs

        return neighbors, edge_idxs, edge_times

    def _sample_indices(self, population_size, n_samples):
        if self.random_state is not None:
            return self.random_state.randint(0, population_size, n_samples)
        return np.random.randint(0, population_size, n_samples)


def cosine_similarity(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())
