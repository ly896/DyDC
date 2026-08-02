import numpy as np
import pandas as pd


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def _load_graph(dataset_name):
    return pd.read_csv(f"./data/ml_{dataset_name}.csv")


def _build_data(graph_df, mask=None):
    if mask is None:
        mask = np.ones(len(graph_df), dtype=bool)

    return Data(
        graph_df.u.values[mask],
        graph_df.i.values[mask],
        graph_df.ts.values[mask],
        graph_df.idx.values[mask],
        graph_df.label.values[mask],
    )


def get_valid_data_node_classification(dataset_name, tuning_ratio=0.7, training_ratio=0.85):
    graph_df = _load_graph(dataset_name)
    val_time, test_time = list(np.quantile(graph_df.ts, [tuning_ratio, training_ratio]))

    timestamps = graph_df.ts.values
    train_mask = timestamps <= val_time
    test_mask = (timestamps > val_time) & (timestamps <= test_time)

    return _build_data(graph_df), _build_data(graph_df, train_mask), _build_data(graph_df, test_mask)


def get_data_node_classification(dataset_name, training_ratio=0.85):
    graph_df = _load_graph(dataset_name)
    test_time = np.quantile(graph_df.ts, training_ratio)

    timestamps = graph_df.ts.values
    train_mask = timestamps <= test_time
    test_mask = timestamps > test_time

    return _build_data(graph_df), _build_data(graph_df, train_mask), _build_data(graph_df, test_mask)
