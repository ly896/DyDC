import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def preprocess(data_name):
    u_list, i_list, ts_list, label_list, idx_list = [], [], [], [], []

    with open(data_name) as file:
        next(file)
        for idx, line in enumerate(file):
            source, destination, timestamp, label = line.strip().split(",")[:4]
            u_list.append(int(source))
            i_list.append(int(destination))
            ts_list.append(float(timestamp))
            label_list.append(float(label))
            idx_list.append(idx)

    return pd.DataFrame(
        {
            "u": u_list,
            "i": i_list,
            "ts": ts_list,
            "label": label_list,
            "idx": idx_list,
        }
    )


def reindex(df, bipartite=True):
    new_df = df.copy()
    if bipartite:
        assert df.u.max() - df.u.min() + 1 == len(df.u.unique())
        assert df.i.max() - df.i.min() + 1 == len(df.i.unique())

        new_df.i = df.i + df.u.max() + 1

    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
    return new_df


def reindex_with_node_id_map(df, bipartite=False):
    new_df = df.copy()
    node_id_map = {}
    counter = 0

    if bipartite:
        n_unique_src = len(set(new_df.u))
        n_unique_dst = len(set(new_df.i))

        for source in new_df.u:
            if source not in node_id_map:
                node_id_map[source] = counter
                counter += 1

        assert counter == n_unique_src
        new_df.i = new_df.i + new_df.u.max() + 1

        for destination in new_df.i:
            if destination not in node_id_map:
                node_id_map[destination] = counter
                counter += 1

        new_df.u = new_df.u.apply(lambda value: node_id_map[value])
        new_df.i = new_df.i.apply(lambda value: node_id_map[value])

        assert new_df.u.max() - new_df.u.min() + 1 == n_unique_src
        assert new_df.i.max() - new_df.i.min() + 1 == n_unique_dst
        assert new_df.u.max() + 1 == new_df.i.min()
    else:
        n_unique_nodes = len(set(new_df.u) | set(new_df.i))
        for row_idx in range(len(new_df.u)):
            source = new_df.u[row_idx]
            destination = new_df.i[row_idx]
            if source not in node_id_map:
                node_id_map[source] = counter
                counter += 1
            if destination not in node_id_map:
                node_id_map[destination] = counter
                counter += 1

        new_df.u = new_df.u.apply(lambda value: node_id_map[value])
        new_df.i = new_df.i.apply(lambda value: node_id_map[value])
        assert counter == n_unique_nodes
        assert len(set(new_df.u) | set(new_df.i)) == n_unique_nodes

    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
    return new_df


def run(data_name, bipartite=True):
    Path("data").mkdir(parents=True, exist_ok=True)
    input_path = f"./data/{data_name}.csv"
    output_df = f"./data/ml_{data_name}.csv"
    output_edge_feat = f"./data/ml_{data_name}.npy"
    output_node_feat = f"./data/ml_{data_name}_node.npy"

    df = preprocess(input_path)
    new_df = reindex_with_node_id_map(df, bipartite)
    max_idx = max(new_df.u.max(), new_df.i.max())
    zero_node_features = np.zeros((max_idx + 1, 172))
    zero_edge_features = np.zeros((len(new_df), 172))

    new_df.to_csv(output_df, index=False)
    np.save(output_edge_feat, zero_edge_features)
    np.save(output_node_feat, zero_node_features)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw graph CSV data.")
    parser.add_argument("--data", type=str, default="wikipedia", help="Raw dataset name.")
    parser.add_argument("--bipartite", action="store_true", help="Whether the graph is bipartite.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data, bipartite=args.bipartite)
