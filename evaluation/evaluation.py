import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def eval_anomaly_node_detection(
    model,
    data,
    batch_size,
    n_neighbors,
    device,
    only_rec_score=False,
    only_drift_score=False,
    test_inference_time=False,
    collect_embeddings=False,
):
    """Evaluate anomaly detection with 1 - cosine similarity as the anomaly score."""
    del only_rec_score, only_drift_score
    pred_score = np.zeros(len(data.sources))
    short_embeddings = [] if collect_embeddings else None
    long_embeddings = [] if collect_embeddings else None
    num_instances = len(data.sources)
    num_batches = math.ceil(num_instances / batch_size)

    with torch.no_grad():
        model.eval()

        for batch_id in range(num_batches):
            start = batch_id * batch_size
            end = min(num_instances, start + batch_size)

            if test_inference_time:
                for idx in range(start, end):
                    source = data.sources[idx]
                    destination = data.destinations[idx]
                    timestamp = data.timestamps[idx]
                    model.neighbor_finder.node_to_neighbors[source] = np.append(
                        model.neighbor_finder.node_to_neighbors[source], destination
                    )
                    model.neighbor_finder.node_to_neighbors[destination] = np.append(
                        model.neighbor_finder.node_to_neighbors[destination], source
                    )
                    model.neighbor_finder.node_to_edge_timestamps[source] = np.append(
                        model.neighbor_finder.node_to_edge_timestamps[source], timestamp
                    )
                    model.neighbor_finder.node_to_edge_timestamps[destination] = np.append(
                        model.neighbor_finder.node_to_edge_timestamps[destination],
                        timestamp,
                    )

            sources_batch = torch.as_tensor(
                data.sources[start:end], dtype=torch.long, device=device
            )
            destinations_batch = torch.as_tensor(
                data.destinations[start:end], dtype=torch.long, device=device
            )
            timestamps_batch = torch.as_tensor(
                data.timestamps[start:end], dtype=torch.float32, device=device
            )

            src_neighbors_np, _, src_neighbors_time_np = model.neighbor_finder.get_temporal_neighbor(
                data.sources[start:end], data.timestamps[start:end], n_neighbors
            )
            dst_neighbors_np, _, dst_neighbors_time_np = model.neighbor_finder.get_temporal_neighbor(
                data.destinations[start:end], data.timestamps[start:end], n_neighbors
            )

            src_neighbors_batch = torch.as_tensor(
                src_neighbors_np, dtype=torch.long, device=device
            )
            dst_neighbors_batch = torch.as_tensor(
                dst_neighbors_np, dtype=torch.long, device=device
            )
            src_neighbors_time_batch = torch.as_tensor(
                src_neighbors_time_np, dtype=torch.float32, device=device
            )
            dst_neighbors_time_batch = torch.as_tensor(
                dst_neighbors_time_np, dtype=torch.float32, device=device
            )

            positive_memory_score, h_short_src, h_long_src, _ = model.compute_anomaly_score(
                sources_batch,
                destinations_batch,
                timestamps_batch,
                src_neighbors_batch,
                dst_neighbors_batch,
                src_neighbors_time_batch,
                dst_neighbors_time_batch,
                n_neighbors,
            )

            similarity = positive_memory_score.reshape(-1).cpu().numpy()
            similarity = np.nan_to_num(similarity)
            pred_score[start:end] = 1.0 - similarity
            if collect_embeddings:
                short_embeddings.append(h_short_src.detach().cpu().numpy())
                long_embeddings.append(h_long_src.detach().cpu().numpy())

    auc_roc = roc_auc_score(data.labels, pred_score)
    embeddings = None
    if collect_embeddings and short_embeddings:
        embeddings = {
            "short": np.concatenate(short_embeddings, axis=0),
            "long": np.concatenate(long_embeddings, axis=0),
        }
    return auc_roc, pred_score, embeddings
