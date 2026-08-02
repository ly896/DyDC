import torch
from torch import nn


class TemporalAttentionLayer(nn.Module):
    """Temporal multi-head attention over sampled historical neighbors."""

    def __init__(
        self,
        n_node_features,
        n_neighbors_features,
        time_dim,
        output_dimension,
        n_head=2,
        dropout=0.1,
    ):
        super().__init__()
        del output_dimension
        self.n_head = n_head
        self.feat_dim = n_node_features
        self.time_dim = time_dim
        self.query_dim = time_dim
        self.key_dim = n_neighbors_features + time_dim
        self.val_dim = n_neighbors_features + time_dim

        self.multi_head_target = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            kdim=self.key_dim,
            vdim=self.val_dim,
            num_heads=n_head,
            dropout=dropout,
        )

    def forward(
        self,
        src_time_features,
        neighbors_features,
        neighbors_time_features,
        neighbors_padding_mask,
    ):
        query = src_time_features.permute(1, 0, 2)
        key = torch.cat([neighbors_features, neighbors_time_features], dim=2).permute(
            1, 0, 2
        )
        value = torch.cat([neighbors_features, neighbors_time_features], dim=2).permute(
            1, 0, 2
        )

        padding_mask = neighbors_padding_mask.clone()
        invalid_neighborhood_mask = padding_mask.all(dim=1, keepdim=True)
        padding_mask[invalid_neighborhood_mask.squeeze(dim=1), 0] = False

        attention_output, attention_weights = self.multi_head_target(
            query=query,
            key=key,
            value=value,
            key_padding_mask=padding_mask,
        )

        attention_output = attention_output.squeeze(0)
        attention_weights = attention_weights.squeeze(1)
        attention_output = attention_output.masked_fill(invalid_neighborhood_mask, 0)
        attention_weights = attention_weights.masked_fill(invalid_neighborhood_mask, 0)
        return attention_output, attention_weights
