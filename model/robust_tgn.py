import copy
import logging

import torch
import torch.nn.functional as F
from torch_scatter import scatter

from model.time_encoding import FixedTimeEncode
from modules.embedding_module import get_embedding_module
from modules.memory import Memory
from modules.memory_updater import get_memory_updater
from modules.message_function import get_message_function


class RobustDynamicGraphAnomalyDetector(torch.nn.Module):
    def __init__(
        self,
        neighbor_finder,
        device,
        n_nodes,
        n_edges,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
        message_dimension=128,
        memory_dimension=256,
        n_neighbors=10,
        memory_agg_type="TGAT",
        message_updater="mlp",
        memory_updater="gated_gru",
        ema_alpha=0.999,
        distill_weight=0.5,
        negative_memory_type="train",
        src_reg_factor=0.1,
        dst_reg_factor=0.1,
        only_drift_loss=False,
        only_recovery_loss=False,
    ):
        super().__init__()
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.n_nodes = n_nodes + 1
        self.n_edges = n_edges
        self.memory_dimension = memory_dimension
        self.time_encoder = FixedTimeEncode(dimension=memory_dimension)

        self.n_node_features = memory_dimension
        self.n_edge_features = memory_dimension
        self.raw_message_dimension = memory_dimension + memory_dimension
        self.message_dimension = message_dimension
        self.memory_agg_type = memory_agg_type
        self.ema_alpha = ema_alpha
        self.distill_weight = distill_weight
        self.negative_memory_type = negative_memory_type
        self.src_reg_factor = src_reg_factor
        self.dst_reg_factor = dst_reg_factor
        self.only_drift_loss = only_drift_loss
        self.only_recovery_loss = only_recovery_loss

        self.memory = Memory(
            n_nodes=self.n_nodes,
            memory_dimension=memory_dimension,
            input_dimension=message_dimension,
            raw_message_dimension=self.raw_message_dimension,
            device=device,
        )

        self.message_function = get_message_function(
            module_type=message_updater,
            raw_message_dimension=self.raw_message_dimension,
            message_dimension=message_dimension,
        )
        self.memory_updater = get_memory_updater(
            module_type=memory_updater,
            memory=self.memory,
            message_dimension=message_dimension,
            memory_dimension=memory_dimension,
            device=device,
        )

        self.embedding_module_student = get_embedding_module(
            module_type="graph_attention_recovery",
            memory=self.memory,
            neighbor_finder=self.neighbor_finder,
            time_encoder=self.time_encoder,
            n_layers=n_layers,
            n_node_features=self.n_node_features,
            n_edge_features=self.n_edge_features,
            n_time_features=memory_dimension,
            embedding_dimension=memory_dimension,
            device=self.device,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=True,
            n_neighbors=n_neighbors,
        )

        self.embedding_module_teacher = copy.deepcopy(self.embedding_module_student)
        for param in self.embedding_module_teacher.parameters():
            param.requires_grad = False

    def update_teacher_parameters(self):
        """Update the teacher encoder with an exponential moving average."""
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.embedding_module_teacher.parameters(),
                self.embedding_module_student.parameters(),
            ):
                teacher_param.mul_(self.ema_alpha).add_(
                    student_param, alpha=1 - self.ema_alpha
                )

    def compute_temporal_embeddings(
        self,
        source_nodes,
        destination_nodes,
        edge_times,
        src_neighbors,
        dst_neighbors,
        src_neighbors_time,
        dst_neighbors_time,
        n_neighbors,
        use_teacher=False,
    ):
        h_long_src = self.memory.get_memory(source_nodes)
        embedding_module = (
            self.embedding_module_teacher if use_teacher else self.embedding_module_student
        )

        nodes = torch.cat([source_nodes, destination_nodes])
        timestamps = torch.cat([edge_times, edge_times])
        neighbors = torch.cat([src_neighbors, dst_neighbors])
        neighbors_time = torch.cat([src_neighbors_time, dst_neighbors_time])

        h_short_all = embedding_module.compute_recovery_memory_embedding(
            memory=self.memory.memory,
            source_nodes=nodes,
            timestamps=timestamps,
            neighbors=neighbors,
            neighbors_time=neighbors_time,
            n_layers=1,
            n_neighbors=n_neighbors,
        )

        h_short_src = h_short_all[: len(source_nodes)]
        h_short_dst = h_short_all[len(source_nodes) :]

        self.update_memory_tensor(
            nodes, self.memory.messages_tensor, self.memory.messages_time
        )
        self.memory.clear_messages(nodes)

        unique_sources, source_messages, source_times = self.get_raw_messages(
            source_nodes, destination_nodes, edge_times
        )
        unique_destinations, destination_messages, destination_times = self.get_raw_messages(
            destination_nodes, source_nodes, edge_times
        )
        self.memory.store_raw_messages(unique_sources, source_messages, source_times)
        self.memory.store_raw_messages(
            unique_destinations, destination_messages, destination_times
        )

        return h_short_src, h_long_src, h_short_dst

    def compute_node_diff_score(
        self,
        source_nodes,
        destination_nodes,
        edge_times,
        src_neighbors,
        dst_neighbors,
        src_neighbors_time,
        dst_neighbors_time,
        n_neighbors,
        seen_nodes,
    ):
        del seen_nodes
        h_short_src, h_long_src, _ = self.compute_temporal_embeddings(
            source_nodes,
            destination_nodes,
            edge_times,
            src_neighbors,
            dst_neighbors,
            src_neighbors_time,
            dst_neighbors_time,
            n_neighbors,
            use_teacher=False,
        )

        with torch.no_grad():
            h_teacher_src = self.embedding_module_teacher.compute_recovery_memory_embedding(
                memory=self.memory.memory,
                source_nodes=source_nodes,
                timestamps=edge_times,
                neighbors=src_neighbors,
                neighbors_time=src_neighbors_time,
                n_layers=1,
                n_neighbors=n_neighbors,
            )

        positive_similarity = F.cosine_similarity(h_short_src, h_long_src)
        contrast_loss = (1 - positive_similarity).mean()
        distill_loss = F.mse_loss(h_short_src, h_teacher_src)
        total_loss = contrast_loss + self.distill_weight * distill_loss
        return positive_similarity, None, None, None, total_loss

    def compute_anomaly_score(
        self,
        source_nodes,
        destination_nodes,
        edge_times,
        src_neighbors,
        dst_neighbors,
        src_neighbors_time,
        dst_neighbors_time,
        n_neighbors,
    ):
        h_short_src, h_long_src, _ = self.compute_temporal_embeddings(
            source_nodes,
            destination_nodes,
            edge_times,
            src_neighbors,
            dst_neighbors,
            src_neighbors_time,
            dst_neighbors_time,
            n_neighbors,
            use_teacher=True,
        )
        score = F.cosine_similarity(h_short_src, h_long_src)
        return score, h_short_src, h_long_src, None

    def get_raw_messages(self, source_nodes, destination_nodes, edge_times):
        destination_memory = self.memory.get_memory(destination_nodes)
        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(
            source_time_delta.unsqueeze(dim=1)
        ).view(len(source_nodes), -1)
        source_message = torch.cat(
            [destination_memory, source_time_delta_encoding], dim=1
        )
        node_ids, inverse_index = torch.unique(source_nodes, return_inverse=True)
        message = scatter(source_message, inverse_index, reduce="mean", dim=0)
        message_ts = scatter(edge_times, inverse_index, reduce="max")
        return node_ids, message, message_ts

    def update_memory_tensor(self, nodes, messages_tensor, messages_ts_tensor):
        unique_node_ids = torch.unique(nodes).to(self.device)
        mask = messages_ts_tensor[unique_node_ids] != 0
        active_nodes = unique_node_ids[mask]
        if len(active_nodes) == 0:
            return

        unique_node_messages = messages_tensor[unique_node_ids][mask]
        unique_node_ts = messages_ts_tensor[unique_node_ids][mask]
        unique_messages = self.message_function.compute_message(unique_node_messages)
        self.memory_updater.update_memory(
            active_nodes, unique_messages, timestamps=unique_node_ts
        )

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module_student.neighbor_finder = neighbor_finder
        self.embedding_module_teacher.neighbor_finder = neighbor_finder
