import torch
import torch.nn.init as init
from torch import nn


class TemporalAttentionLayer(nn.Module):
    """Custom temporal attention layer matching DySAT's implementation exactly.

    Key differences from PyTorch's MultiheadAttention:
    - Scales by sqrt(num_time_steps) instead of sqrt(d_k)
    - Matches DySAT's exact architecture
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_time_steps: int,
        dropout: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_time_steps = num_time_steps  # Maximum number of time steps
        self.head_dim = embed_dim // num_heads
        # Q, K, V projection matrices (matching DySAT)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        init.xavier_uniform_(self.q_proj.weight)
        init.xavier_uniform_(self.k_proj.weight)
        init.xavier_uniform_(self.v_proj.weight)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [N, T, F] tensor where N=num_nodes, T=num_time_steps, F=embed_dim
            attn_mask: [T, T] boolean mask (True = mask out, False = keep)

        Returns:
            [N, T, F] output tensor
        """
        # [N, T, F] where N=num_nodes, T=num_time_steps, F=embed_dim
        N, T, F = x.shape
        assert F == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {F}"

        # [N, T, F] - Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [N, T, F] -> [N, T, num_heads, head_dim] -> [N*num_heads, T, head_dim]
        # Reshape for multi-head attention
        q = q.view(N, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # [hN, T, F/h]
        q = q.view(N * self.num_heads, T, self.head_dim)
        k = k.view(N, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # [hN, T, F/h]
        k = k.view(N * self.num_heads, T, self.head_dim)
        v = v.view(N, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # [hN, T, F/h]
        v = v.view(N * self.num_heads, T, self.head_dim)

        # Scale by sqrt(num_time_steps) as in DySAT (not sqrt(head_dim))
        scale = self.num_time_steps**0.5
        # [hN, T, T] - Attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / scale

        if attn_mask is not None:
            # [T, T] - attn_mask with True=mask, False=keep
            # Expand to [hN, T, T]
            attn_mask_expanded = attn_mask.unsqueeze(0).expand(
                N * self.num_heads, -1, -1
            )
            attn_scores = attn_scores.masked_fill(attn_mask_expanded, float("-inf"))

        # [hN, T, T] - Attention weights after softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # [hN, T, F/h] - Attention output
        attn_output = torch.bmm(attn_weights, v)

        # [hN, T, F/h] -> [N, num_heads, T, F/h] -> [N, T, num_heads, F/h] -> [N, T, F]
        # Reshape back and concatenate heads
        attn_output = attn_output.view(N, self.num_heads, T, self.head_dim)
        # [N, T, num_heads, F/h] - Transposed for concatenation
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [N, T, F] - Concatenated heads (DySAT doesn't have output projection)
        attn_output = attn_output.view(N, T, F)

        return attn_output


class TemporalBlock(nn.Module):
    def __init__(
        self,
        edim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        num_ss: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.pos_emb = nn.Embedding(num_ss, edim)
        init.xavier_uniform_(self.pos_emb.weight)
        self.attn_layers = nn.ModuleList(
            [
                TemporalAttentionLayer(
                    embed_dim=edim,
                    num_heads=heads,
                    num_time_steps=num_ss,  # Maximum number of time steps
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.feedforward = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(edim, edim),
                    nn.ReLU(),
                )
                for _ in range(num_layers)
            ]
        )

        self.embed_dim = edim
        self.num_blocks = num_layers
        self.heads = heads
        self.dropout = dropout
        self.num_ss = num_ss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, T, F] where N=num_nodes, T=num_snapshots, F=embed_dim
        N, T, F = x.shape
        assert F == self.embed_dim, "embed_dim mismatch!"
        assert T <= self.num_ss, (
            f"Number of snapshots ({T}) exceeds embedding table size ({self.num_ss})!"
        )

        # [N, T] - Position indices for each node
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(N, T)
        # [N, T, F] - Add position embeddings to input
        temporal_inputs = x + self.pos_emb(positions)
        # [T, T] - Causal mask (upper triangular, True=mask)
        attn_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        x = temporal_inputs
        for attn, ff in zip(self.attn_layers, self.feedforward):
            # [N, T, F] - Temporal attention output
            y = attn(x, attn_mask=attn_mask)
            # [N, T, F] - Feedforward output
            ff_out = ff(y)
            # [N, T, F] - Residual connection
            y = ff_out + y
            x = y
        # [N, T, F] - Final output
        return y

    def get_grads(self):
        model_parameters = list(self.parameters())
        grads = [parameter.grad for parameter in model_parameters]
        return grads

    def set_grads(self, grads):
        model_parameters = list(self.parameters())
        for grad, parameter in zip(grads, model_parameters):
            parameter.grad = grad
