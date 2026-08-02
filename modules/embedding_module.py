import torch
from torch import nn

from model.temporal_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
    def __init__(
        self,
        memory,
        neighbor_finder,
        time_encoder,
        n_layers,
        n_node_features,
        n_edge_features,
        n_time_features,
        embedding_dimension,
        device,
        dropout,
    ):
        super().__init__()
        self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20):
        raise NotImplementedError


class GraphEmbedding(EmbeddingModule):
    def __init__(
        self,
        memory,
        neighbor_finder,
        time_encoder,
        n_layers,
        n_node_features,
        n_edge_features,
        n_time_features,
        embedding_dimension,
        device,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
    ):
        super().__init__(
            memory,
            neighbor_finder,
            time_encoder,
            n_layers,
            n_node_features,
            n_edge_features,
            n_time_features,
            embedding_dimension,
            device,
            dropout,
        )
        del n_heads
        self.use_memory = use_memory
        self.memory_dropout = dropout

    def compute_recovery_memory_embedding(
        self,
        memory,
        source_nodes,
        timestamps,
        neighbors,
        neighbors_time,
        n_layers,
        n_neighbors=20,
    ):
        assert n_layers >= 0
        timestamps_torch = timestamps.unsqueeze(dim=1)
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))

        neighbors_torch = neighbors
        edge_deltas_torch = timestamps.unsqueeze(dim=1) - neighbors_time
        flattened_neighbors = neighbors.flatten()

        neighbor_embeddings = memory[flattened_neighbors, :]
        effective_neighbors = n_neighbors if n_neighbors > 0 else 1
        neighbor_embeddings = neighbor_embeddings.view(
            len(source_nodes), effective_neighbors, -1
        )

        edge_time_embeddings = self.time_encoder(edge_deltas_torch)
        padding_mask = neighbors_torch == 0

        if self.training:
            memory_mask = torch.rand(padding_mask.size(), device=self.device) < self.memory_dropout
            padding_mask = torch.logical_or(padding_mask, memory_mask)

        return self.aggregate(
            n_layers,
            source_nodes_time_embedding,
            neighbor_embeddings,
            edge_time_embeddings,
            padding_mask,
        )

    def aggregate(self, n_layers, source_nodes_time_embedding, neighbor_embeddings, edge_time_embeddings, mask):
        raise NotImplementedError


class GraphAttentionEmbeddingRecovery(GraphEmbedding):
    def __init__(
        self,
        memory,
        neighbor_finder,
        time_encoder,
        n_layers,
        n_node_features,
        n_edge_features,
        n_time_features,
        embedding_dimension,
        device,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
    ):
        super().__init__(
            memory,
            neighbor_finder,
            time_encoder,
            n_layers,
            n_node_features,
            n_edge_features,
            n_time_features,
            embedding_dimension,
            device,
            n_heads,
            dropout,
            use_memory,
        )

        self.attention_model = TemporalAttentionLayer(
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            time_dim=n_time_features,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=n_node_features,
        )

    def aggregate(
        self,
        n_layers,
        source_nodes_time_embedding,
        neighbor_embeddings,
        edge_time_embeddings,
        mask,
    ):
        del n_layers
        source_embedding, _ = self.attention_model(
            source_nodes_time_embedding,
            neighbor_embeddings,
            edge_time_embeddings,
            mask,
        )
        return source_embedding


def get_embedding_module(
    module_type,
    memory,
    neighbor_finder,
    time_encoder,
    n_layers,
    n_node_features,
    n_edge_features,
    n_time_features,
    embedding_dimension,
    device,
    n_heads=2,
    dropout=0.1,
    n_neighbors=None,
    use_memory=True,
):
    del n_neighbors
    if module_type == "graph_attention_recovery":
        return GraphAttentionEmbeddingRecovery(
            memory=memory,
            neighbor_finder=neighbor_finder,
            time_encoder=time_encoder,
            n_layers=n_layers,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_time_features=n_time_features,
            embedding_dimension=embedding_dimension,
            device=device,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=use_memory,
        )
    raise ValueError(f"Embedding module {module_type} is not supported.")
