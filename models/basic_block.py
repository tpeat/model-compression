import torch
from torch import nn

# Define the Vision Transformer Block
class ViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super(ViTBlock, self).__init__()
        self.dim = dim
        self.heads = heads
        self.mlp_dim = mlp_dim

        # Multi-head Self Attention
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

        # Layer norm layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm2(x)
        x = x + self.ffn(x)
        return x