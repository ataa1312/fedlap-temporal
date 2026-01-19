import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn as nn
from torch_geometric.utils import scatter, softmax
from torch_geometric.typing import Adj, OptTensor


class CustomGAT(nn.Module):
    """Single structural attention head matching DySAT's implementation exactly.

    This implements the concatenation-based self-attention mechanism from DySAT:
    - Concatenates transformed features for source and target nodes
    - Formula: e_ij = LeakyReLU(a^T [W^s x_i || W^s x_j]) where || denotes concatenation
    - Applies separate dropouts for attention coefficients and features
    """

    def __init__(
        self,
        din: int,
        dout: int,
        attention_dropout: float = 0.0,
        feedforward_dropout: float = 0.0,
        residual: bool = False,
        leaky_relu_alpha: float = 0.2,
    ):
        super().__init__()
        self.din = din
        self.dout = dout
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.residual = residual
        self.leaky_relu_alpha = leaky_relu_alpha

        self.weight_transform = nn.Linear(din, dout, bias=False)
        self.attn_vector = nn.Linear(2 * dout, 1, bias=False)  # Attention vector a^T
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_alpha)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.feature_dropout = nn.Dropout(feedforward_dropout)
        if residual:
            self.residual_transform = nn.Linear(din, dout, bias=False)
        else:
            self.residual_transform = None

        init.xavier_uniform_(self.weight_transform.weight)
        init.xavier_uniform_(self.attn_vector.weight)
        if self.residual_transform is not None:
            init.xavier_uniform_(self.residual_transform.weight)

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight_transform.weight)
        init.xavier_uniform_(self.attn_vector.weight)
        if self.residual_transform is not None:
            init.xavier_uniform_(self.residual_transform.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, din] node features
            edge_index: [2, E] edge indices
            edge_weight: [E] optional edge weights

        Returns:
            [N, dout] output features
        """
        num_nodes = x.size(0)
        # seq_feats.shape =  [num_nodes, out_dim]
        seq_feats = self.weight_transform(x)
        if isinstance(edge_index, torch.Tensor):
            row = edge_index[0]
            col = edge_index[1]
        else:
            row, col, _ = edge_index.coo()

        # src_feats.shape = [num_edges, dout]
        # tgt_feats.shape = [num_edges, dout]
        src_feats = seq_feats[row]
        tgt_feats = seq_feats[col]
        # concat_feats.shape = [num_edges, 2 * dout] (concatenated along feature dimension)
        concat_fts = torch.cat([src_feats, tgt_feats], dim=-1)

        # logits.shape = [num_edges, 1]
        logits = self.attn_vector(concat_fts)
        # logits.shape = [num_edges]
        logits = logits.squeeze(-1)
        if edge_weight is not None:
            logits = logits * edge_weight
        logits = self.leaky_relu(logits)
        attn_coeffs = softmax(logits, row, num_nodes=num_nodes)

        # Apply attention coefficient dropout (after softmax, matching DySAT)
        if self.training and self.attention_dropout > 0:
            attn_coeffs = self.attn_dropout(attn_coeffs)

        # Apply feature dropout (before aggregation, matching DySAT)
        if self.training and self.feedforward_dropout > 0:
            seq_feats = self.feature_dropout(seq_feats)

        # output.shape = [N, out_dim]
        output = scatter(
            src=attn_coeffs.unsqueeze(-1) * seq_feats[col],
            index=row,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        if self.residual and self.residual_transform is not None:
            output = output + self.residual_transform(x)

        return output


class MultiHeadCustomGAT(nn.Module):
    """Multi-head custom GAT layer similar to StructuralBlock.

    This wraps multiple CustomGAT heads and concatenates their outputs,
    making it compatible with the GNN class's layer-by-layer API.
    """

    def __init__(
        self,
        din: int,
        dout: int,
        num_heads: int,
        per_head_out_dim: int,
        attention_dropout: float = 0.0,
        feedforward_dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        self.din = din
        self.dout = dout
        self.num_heads = num_heads
        self.per_head_out_dim = per_head_out_dim

        # Create multiple attention heads
        self.heads = nn.ModuleList(
            [
                CustomGAT(
                    din=din,
                    dout=per_head_out_dim,
                    attention_dropout=attention_dropout,
                    feedforward_dropout=feedforward_dropout,
                    residual=residual,
                )
                for _ in range(num_heads)
            ]
        )

    def reset_parameters(self) -> None:
        for head in self.heads:
            head.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, din] node features
            edge_index: [2, E] edge indices
            edge_weight: [E] optional edge weights

        Returns:
            [N, dout] output features (concatenated from all heads)
        """
        head_outputs = []
        for head in self.heads:
            head_out = head(x, edge_index, edge_weight)
            head_outputs.append(head_out)

        # Concatenate all head outputs
        x = torch.cat(head_outputs, dim=-1)
        # Apply ELU activation (matching StructuralBlock)
        x = F.elu(x)

        return x
