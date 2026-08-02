# import argparse
# import logging
# import math
# import random
# import sys
# from pathlib import Path
#
# import numpy as np
# import torch
# from tqdm import tqdm
#
# from evaluation.evaluation import eval_anomaly_node_detection
# from model.robust_tgn import RobustDynamicGraphAnomalyDetector
# from utils.data_processing import get_data_node_classification
# from utils.utils import get_neighbor_finder
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Train a robust dynamic graph anomaly detection model."
#     )
#     parser.add_argument(
#         "-d",
#         "--data",
#         type=str,
#         default="wikipedia",
#         help="Dataset name under data/, without the ml_ prefix or .csv suffix.",
#     )
#     parser.add_argument("--bs", type=int, default=100, help="Batch size.")
#     parser.add_argument("--n_degree", type=int, default=20, help="Number of temporal neighbors.")
#     parser.add_argument("--n_head", type=int, default=2, help="Number of attention heads.")
#     parser.add_argument("--n_epoch", type=int, default=20, help="Number of training epochs.")
#     parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate.")
#     parser.add_argument("--n_runs", type=int, default=1, help="Number of repeated runs.")
#     parser.add_argument("--seed", type=int, default=0, help="Random seed.")
#     parser.add_argument("--drop_out", type=float, default=0.1, help="Dropout probability.")
#     parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
#     parser.add_argument("--message_dim", type=int, default=128, help="Message dimension.")
#     parser.add_argument("--memory_dim", type=int, default=256, help="Memory dimension.")
#     parser.add_argument("--agg_type", type=str, default="TGAT", help="Memory recovery aggregation type.")
#     parser.add_argument(
#         "--negative_memory_type",
#         type=str,
#         default="train",
#         choices=["train", "random"],
#         help="Negative memory sampling strategy.",
#     )
#     parser.add_argument(
#         "--message_updater",
#         type=str,
#         default="mlp",
#         choices=["mlp", "identity"],
#         help="Message function type.",
#     )
#     parser.add_argument(
#         "--memory_updater",
#         type=str,
#         default="gated_gru",
#         choices=["gru", "rnn", "gated_gru"],
#         help="Memory updater type.",
#     )
#     parser.add_argument("--training_ratio", type=float, default=0.90, help="Training split ratio.")
#     parser.add_argument("--lr_decay", type=float, default=0.8, help="Learning-rate decay factor.")
#     parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
#     parser.add_argument("--srf", type=float, default=0.1, help="Source contrastive regularization weight.")
#     parser.add_argument("--drf", type=float, default=0.1, help="Destination contrastive regularization weight.")
#     parser.add_argument("--distill_weight", type=float, default=0.5, help="Self-distillation loss weight.")
#     parser.add_argument("--ema_alpha", type=float, default=0.999, help="Teacher EMA update factor.")
#     parser.add_argument("--only_drift_loss_score", action="store_true", help="Use drift loss and score only.")
#     parser.add_argument("--only_recovery_loss_score", action="store_true", help="Use recovery loss and score only.")
#     parser.add_argument("--only_drift_score", action="store_true", help="Use drift score and both losses.")
#     parser.add_argument("--only_rec_score", action="store_true", help="Use recovery score and both losses.")
#     parser.add_argument("--test_inference_time", action="store_true", help="Update graph history during testing.")
#     return parser.parse_args()
#
#
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#
#
# def setup_logging():
#     Path("log").mkdir(parents=True, exist_ok=True)
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )
#     return logging.getLogger(__name__)
#
#
# def to_tensor(array, dtype, device):
#     return torch.as_tensor(array, dtype=dtype, device=device)
#
#
# def main():
#     args = parse_args()
#     set_seed(args.seed)
#     logger = setup_logging()
#
#     batch_size = args.bs
#     num_neighbors = args.n_degree
#     num_epochs = args.n_epoch
#     num_heads = args.n_head
#     num_layers = 1
#
#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#     logger.info("Arguments: %s", args)
#     logger.info("Using device: %s", device)
#
#     full_data, train_data, test_data = get_data_node_classification(
#         args.data, training_ratio=args.training_ratio
#     )
#
#     max_node_idx = max(full_data.unique_nodes)
#     train_ngh_finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_node_idx)
#     full_ngh_finder = get_neighbor_finder(full_data, uniform=False, max_node_idx=max_node_idx)
#
#     logger.info("Pre-sampling temporal neighbors for training data.")
#     src_neighbors, _, src_neighbors_time = train_ngh_finder.get_temporal_neighbor_tqdm(
#         train_data.sources, train_data.timestamps, num_neighbors
#     )
#     dst_neighbors, _, dst_neighbors_time = train_ngh_finder.get_temporal_neighbor_tqdm(
#         train_data.destinations, train_data.timestamps, num_neighbors
#     )
#
#     test_aucs = []
#     for run_id in range(args.n_runs):
#         logger.info("Run %d/%d started.", run_id + 1, args.n_runs)
#
#         model = RobustDynamicGraphAnomalyDetector(
#             neighbor_finder=train_ngh_finder,
#             n_nodes=full_data.n_unique_nodes,
#             n_edges=full_data.n_interactions,
#             device=device,
#             n_layers=num_layers,
#             n_heads=num_heads,
#             dropout=args.drop_out,
#             message_dimension=args.message_dim,
#             memory_dimension=args.memory_dim,
#             n_neighbors=num_neighbors,
#             memory_agg_type=args.agg_type,
#             negative_memory_type=args.negative_memory_type,
#             message_updater=args.message_updater,
#             memory_updater=args.memory_updater,
#             src_reg_factor=args.srf,
#             dst_reg_factor=args.drf,
#             only_drift_loss=args.only_drift_loss_score,
#             only_recovery_loss=args.only_recovery_loss_score,
#             ema_alpha=args.ema_alpha,
#             distill_weight=args.distill_weight,
#         ).to(device)
#
#         train_sources = to_tensor(train_data.sources, torch.long, device)
#         train_destinations = to_tensor(train_data.destinations, torch.long, device)
#         train_timestamps = to_tensor(train_data.timestamps, torch.float32, device)
#         train_src_neighbors = to_tensor(src_neighbors, torch.long, device)
#         train_dst_neighbors = to_tensor(dst_neighbors, torch.long, device)
#         train_src_neighbors_time = to_tensor(src_neighbors_time, torch.float32, device)
#         train_dst_neighbors_time = to_tensor(dst_neighbors_time, torch.float32, device)
#
#         num_instances = len(train_data.sources)
#         num_batches = math.ceil(num_instances / batch_size)
#
#         optimizer = torch.optim.Adam(
#             model.parameters(), lr=args.lr, weight_decay=args.weight_decay
#         )
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(
#             optimizer, gamma=args.lr_decay
#         )
#         negative_train_nodes = to_tensor(
#             np.array(list(set(train_data.destinations) | set(train_data.sources))),
#             torch.long,
#             device,
#         )
#
#         val_auc = float("nan")
#         for epoch in range(num_epochs):
#             model.memory.__init_memory__()
#             model.set_neighbor_finder(train_ngh_finder)
#             epoch_losses = []
#
#             model.train()
#             for batch_id in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{num_epochs}"):
#                 optimizer.zero_grad()
#                 start = batch_id * batch_size
#                 end = min(num_instances, start + batch_size)
#
#                 _, _, _, _, total_loss = model.compute_node_diff_score(
#                     train_sources[start:end],
#                     train_destinations[start:end],
#                     train_timestamps[start:end],
#                     train_src_neighbors[start:end],
#                     train_dst_neighbors[start:end],
#                     train_src_neighbors_time[start:end],
#                     train_dst_neighbors_time[start:end],
#                     num_neighbors,
#                     negative_train_nodes,
#                 )
#
#                 total_loss.backward()
#                 optimizer.step()
#                 model.update_teacher_parameters()
#                 model.memory.detach_memory()
#                 epoch_losses.append(total_loss.item())
#
#             scheduler.step()
#             model.set_neighbor_finder(full_ngh_finder)
#
#             val_auc, _, _ = eval_anomaly_node_detection(
#                 model,
#                 test_data,
#                 batch_size,
#                 n_neighbors=num_neighbors,
#                 device=device,
#                 test_inference_time=args.test_inference_time,
#             )
#
#             logger.info(
#                 "Epoch %d - mean loss: %.4f - test AUC: %.4f",
#                 epoch + 1,
#                 float(np.mean(epoch_losses)),
#                 val_auc,
#             )
#
#         test_aucs.append(val_auc)
#
#     logger.info("Final mean AUC: %.4f", float(np.mean(test_aucs)))
#
#
# if __name__ == "__main__":
#     main()

import argparse
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from evaluation.evaluation import eval_anomaly_node_detection
from model.robust_tgn import RobustDynamicGraphAnomalyDetector
from utils.data_processing import get_data_node_classification
from utils.tsne_visualization import plot_tsne_embeddings
from utils.utils import get_neighbor_finder

# 尝试导入 psutil 用于监控系统/CPU 内存占用
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a robust dynamic graph anomaly detection model."
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="bitcoinotc",
        help="Dataset name under data/, without the ml_ prefix or .csv suffix.",
    )
    parser.add_argument("--bs", type=int, default=100, help="Batch size.")
    parser.add_argument("--n_degree", type=int, default=20, help="Number of temporal neighbors.")
    parser.add_argument("--n_head", type=int, default=2, help="Number of attention heads.")
    parser.add_argument("--n_epoch", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate.")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of repeated runs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--drop_out", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument("--message_dim", type=int, default=128, help="Message dimension.")
    parser.add_argument("--memory_dim", type=int, default=256, help="Memory dimension.")
    parser.add_argument("--agg_type", type=str, default="TGAT", help="Memory recovery aggregation type.")
    parser.add_argument(
        "--negative_memory_type",
        type=str,
        default="train",
        choices=["train", "random"],
        help="Negative memory sampling strategy.",
    )
    parser.add_argument(
        "--message_updater",
        type=str,
        default="mlp",
        choices=["mlp", "identity"],
        help="Message function type.",
    )
    parser.add_argument(
        "--memory_updater",
        type=str,
        default="gated_gru",
        choices=["gru", "rnn", "gated_gru"],
        help="Memory updater type.",
    )
    parser.add_argument("--training_ratio", type=float, default=0.90, help="Training split ratio.")
    parser.add_argument("--lr_decay", type=float, default=0.8, help="Learning-rate decay factor.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--srf", type=float, default=0.1, help="Source contrastive regularization weight.")
    parser.add_argument("--drf", type=float, default=0.1, help="Destination contrastive regularization weight.")
    parser.add_argument("--distill_weight", type=float, default=0.5, help="Self-distillation loss weight.")
    parser.add_argument("--ema_alpha", type=float, default=0.999, help="Teacher EMA update factor.")
    parser.add_argument("--only_drift_loss_score", action="store_true", help="Use drift loss and score only.")
    parser.add_argument("--only_recovery_loss_score", action="store_true", help="Use recovery loss and score only.")
    parser.add_argument("--only_drift_score", action="store_true", help="Use drift score and both losses.")
    parser.add_argument("--only_rec_score", action="store_true", help="Use recovery score and both losses.")
    parser.add_argument("--test_inference_time", action="store_true", help="Update graph history during testing.")
    parser.add_argument("--tsne", action="store_true", help="Save a t-SNE visualization for the final test embeddings.")
    parser.add_argument(
        "--tsne_feature",
        type=str,
        default="diff",
        choices=["short", "long", "diff", "concat"],
        help="Embedding representation used by t-SNE.",
    )
    parser.add_argument(
        "--tsne_max_points",
        type=int,
        default=2000,
        help="Maximum number of test interactions sampled for t-SNE. Use <=0 for all points.",
    )
    parser.add_argument(
        "--tsne_output_dir",
        type=str,
        default="figure/tsne",
        help="Directory for saved t-SNE PNG and NPZ files.",
    )
    parser.add_argument("--tsne_perplexity", type=float, default=80.0, help="t-SNE perplexity.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging():
    Path("log").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def to_tensor(array, dtype, device):
    return torch.as_tensor(array, dtype=dtype, device=device)


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logging()

    batch_size = args.bs
    num_neighbors = args.n_degree
    num_epochs = args.n_epoch
    num_heads = args.n_head
    num_layers = 1

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("Arguments: %s", args)
    logger.info("Using device: %s", device)

    full_data, train_data, test_data = get_data_node_classification(
        args.data, training_ratio=args.training_ratio
    )

    max_node_idx = max(full_data.unique_nodes)
    train_ngh_finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_node_idx)
    full_ngh_finder = get_neighbor_finder(full_data, uniform=False, max_node_idx=max_node_idx)

    logger.info("Pre-sampling temporal neighbors for training data.")
    src_neighbors, _, src_neighbors_time = train_ngh_finder.get_temporal_neighbor_tqdm(
        train_data.sources, train_data.timestamps, num_neighbors
    )
    dst_neighbors, _, dst_neighbors_time = train_ngh_finder.get_temporal_neighbor_tqdm(
        train_data.destinations, train_data.timestamps, num_neighbors
    )

    test_aucs = []
    for run_id in range(args.n_runs):
        logger.info("Run %d/%d started.", run_id + 1, args.n_runs)

        model = RobustDynamicGraphAnomalyDetector(
            neighbor_finder=train_ngh_finder,
            n_nodes=full_data.n_unique_nodes,
            n_edges=full_data.n_interactions,
            device=device,
            n_layers=num_layers,
            n_heads=num_heads,
            dropout=args.drop_out,
            message_dimension=args.message_dim,
            memory_dimension=args.memory_dim,
            n_neighbors=num_neighbors,
            memory_agg_type=args.agg_type,
            negative_memory_type=args.negative_memory_type,
            message_updater=args.message_updater,
            memory_updater=args.memory_updater,
            src_reg_factor=args.srf,
            dst_reg_factor=args.drf,
            only_drift_loss=args.only_drift_loss_score,
            only_recovery_loss=args.only_recovery_loss_score,
            ema_alpha=args.ema_alpha,
            distill_weight=args.distill_weight,
        ).to(device)

        train_sources = to_tensor(train_data.sources, torch.long, device)
        train_destinations = to_tensor(train_data.destinations, torch.long, device)
        train_timestamps = to_tensor(train_data.timestamps, torch.float32, device)
        train_src_neighbors = to_tensor(src_neighbors, torch.long, device)
        train_dst_neighbors = to_tensor(dst_neighbors, torch.long, device)
        train_src_neighbors_time = to_tensor(src_neighbors_time, torch.float32, device)
        train_dst_neighbors_time = to_tensor(dst_neighbors_time, torch.float32, device)

        num_instances = len(train_data.sources)
        num_batches = math.ceil(num_instances / batch_size)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_decay
        )
        negative_train_nodes = to_tensor(
            np.array(list(set(train_data.destinations) | set(train_data.sources))),
            torch.long,
            device,
        )

        val_auc = float("nan")
        for epoch in range(num_epochs):
            model.memory.__init_memory__()
            model.set_neighbor_finder(train_ngh_finder)
            epoch_losses = []

            # 在训练前重置 GPU 峰值内存统计
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            epoch_start_time = time.perf_counter()

            model.train()
            for batch_id in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{num_epochs}"):
                optimizer.zero_grad()
                start = batch_id * batch_size
                end = min(num_instances, start + batch_size)

                _, _, _, _, total_loss = model.compute_node_diff_score(
                    train_sources[start:end],
                    train_destinations[start:end],
                    train_timestamps[start:end],
                    train_src_neighbors[start:end],
                    train_dst_neighbors[start:end],
                    train_src_neighbors_time[start:end],
                    train_dst_neighbors_time[start:end],
                    num_neighbors,
                    negative_train_nodes,
                )

                total_loss.backward()
                optimizer.step()
                model.update_teacher_parameters()
                model.memory.detach_memory()
                epoch_losses.append(total_loss.item())

            # 仅记录训练阶段的耗时
            epoch_train_time = time.perf_counter() - epoch_start_time

            # 统计内存占用情况 (单位: MB)
            gpu_mem_allocated = 0.0
            gpu_mem_max = 0.0
            if device.type == "cuda":
                gpu_mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
                gpu_mem_max = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            cpu_mem_rss = 0.0
            if HAS_PSUTIL:
                process = psutil.Process(os.getpid())
                cpu_mem_rss = process.memory_info().rss / (1024 ** 2)

            scheduler.step()
            model.set_neighbor_finder(full_ngh_finder)

            collect_tsne_embeddings = args.tsne and epoch == num_epochs - 1
            val_auc, pred_score, embeddings = eval_anomaly_node_detection(
                model,
                test_data,
                batch_size,
                n_neighbors=num_neighbors,
                device=device,
                test_inference_time=args.test_inference_time,
                collect_embeddings=collect_tsne_embeddings,
            )

            if collect_tsne_embeddings and embeddings is not None:
                tsne_prefix = f"{args.data}_run{run_id + 1}_epoch{epoch + 1}"
                tsne_paths = plot_tsne_embeddings(
                    embeddings=embeddings,
                    labels=test_data.labels,
                    scores=pred_score,
                    output_dir=args.tsne_output_dir,
                    prefix=tsne_prefix,
                    feature=args.tsne_feature,
                    max_points=args.tsne_max_points,
                    random_state=args.seed + run_id,
                    perplexity=args.tsne_perplexity,
                    title=f"{args.data} run {run_id + 1} test t-SNE ({args.tsne_feature})",
                    logger=logger,
                )
                if tsne_paths is not None:
                    logger.info(
                        "Saved t-SNE visualization to %s and %s",
                        tsne_paths["image"],
                        tsne_paths["data"],
                    )

            # 输出更丰富的日志信息
            if device.type == "cuda":
                logger.info(
                    "Epoch %d - mean loss: %.4f - test AUC: %.4f - time: %.2fs - GPU Mem (curr/max): %.1fMB/%.1fMB",
                    epoch + 1,
                    float(np.mean(epoch_losses)),
                    val_auc,
                    epoch_train_time,
                    gpu_mem_allocated,
                    gpu_mem_max,
                )
            else:
                if HAS_PSUTIL:
                    logger.info(
                        "Epoch %d - mean loss: %.4f - test AUC: %.4f - time: %.2fs - CPU Mem: %.1fMB",
                        epoch + 1,
                        float(np.mean(epoch_losses)),
                        val_auc,
                        epoch_train_time,
                        cpu_mem_rss,
                    )
                else:
                    logger.info(
                        "Epoch %d - mean loss: %.4f - test AUC: %.4f - time: %.2fs",
                        epoch + 1,
                        float(np.mean(epoch_losses)),
                        val_auc,
                        epoch_train_time,
                    )

        test_aucs.append(val_auc)

    logger.info("Final mean AUC: %.4f", float(np.mean(test_aucs)))


if __name__ == "__main__":
    main()
