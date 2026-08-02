from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def _select_embedding(embeddings, feature):
    short_embedding = embeddings["short"]
    long_embedding = embeddings["long"]
    if feature == "short":
        return short_embedding
    if feature == "long":
        return long_embedding
    if feature == "diff":
        return short_embedding - long_embedding
    if feature == "concat":
        return np.concatenate([short_embedding, long_embedding], axis=1)
    raise ValueError(f"Unsupported t-SNE feature: {feature}")


def _sample_points(features, labels, scores, max_points, random_state):
    num_points = len(labels)
    if max_points is None or max_points <= 0 or num_points <= max_points:
        indices = np.arange(num_points)
    else:
        rng = np.random.RandomState(random_state)
        indices_by_class = [np.where(labels == label)[0] for label in np.unique(labels)]
        sampled_indices = []
        remaining = max_points

        for class_id, class_indices in enumerate(indices_by_class):
            classes_left = len(indices_by_class) - class_id
            target = int(round(max_points * len(class_indices) / num_points))
            target = max(1, min(len(class_indices), target, remaining - classes_left + 1))
            sampled_indices.append(
                rng.choice(class_indices, size=target, replace=False)
            )
            remaining -= target

        indices = np.concatenate(sampled_indices)
        if len(indices) < max_points:
            unused = np.setdiff1d(np.arange(num_points), indices, assume_unique=False)
            extra_size = min(max_points - len(indices), len(unused))
            if extra_size > 0:
                indices = np.concatenate(
                    [indices, rng.choice(unused, size=extra_size, replace=False)]
                )
        rng.shuffle(indices)

    sampled_scores = None if scores is None else scores[indices]
    return features[indices], labels[indices], sampled_scores, indices


def _standardize(features):
    features = np.nan_to_num(features)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (features - mean) / std


def _run_tsne(features, random_state, perplexity):
    perplexity = min(perplexity, max(1, (len(features) - 1) // 3))
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=random_state,
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            n_iter=1000,
            random_state=random_state,
        )
    return tsne.fit_transform(features)


def plot_tsne_embeddings(
    embeddings,
    labels,
    scores,
    output_dir,
    prefix,
    feature="diff",
    max_points=2000,
    random_state=0,
    perplexity=30,
    title=None,
    logger=None,
):
    labels = np.asarray(labels)
    scores = None if scores is None else np.asarray(scores)
    features = _select_embedding(embeddings, feature)
    features, labels, scores, indices = _sample_points(
        features, labels, scores, max_points, random_state
    )

    if len(features) < 4:
        if logger is not None:
            logger.warning("Skip t-SNE: need at least 4 samples, got %d.", len(features))
        return None

    features = _standardize(features)
    coords = _run_tsne(features, random_state=random_state, perplexity=perplexity)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{prefix}_{feature}_tsne.png"
    data_path = output_dir / f"{prefix}_{feature}_tsne.npz"

    np.savez(
        data_path,
        coords=coords,
        labels=labels,
        scores=scores,
        sampled_indices=indices,
        feature=feature,
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    normal_mask = labels == 0
    anomaly_mask = labels != 0

    if normal_mask.any():
        ax.scatter(
            coords[normal_mask, 0],
            coords[normal_mask, 1],
            s=15,
            c="#2A9D8F",
            alpha=0.72,
            linewidths=0,
            label=f"Normal ({normal_mask.sum()})",
        )
    if anomaly_mask.any():
        ax.scatter(
            coords[anomaly_mask, 0],
            coords[anomaly_mask, 1],
            s=22,
            c="#D1495B",
            alpha=0.82,
            linewidths=0,
            label=f"Anomaly ({anomaly_mask.sum()})",
        )

    ax.set_title(title or f"t-SNE of {feature} embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False)
    ax.grid(True, color="#E6E8EB", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"image": image_path, "data": data_path}
