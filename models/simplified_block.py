# Improvements from Simplified Tranformer Blocks
# https://arxiv.org/pdf/2311.01906.pdf

import torch
from torch import nn

class ShapedAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        # Assuming 'dim' is fully divisible by 'heads'
        self.head_dim = dim // heads
        self.heads = heads
        self.scale = self.head_dim ** -0.5

        # Define the Q, K, V projections for all heads at once
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        # self.value = nn.Linear(dim, dim)

        # Output projection
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _ = x.shape

        # Split the Q, K, V matrices into multiple heads and scale query
        q = self.query(x).view(b, n, self.heads, self.head_dim).transpose(1, 2) * self.scale
        k = self.key(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        # removed any learnable params from value
        v = x.view(b, n, self.heads, self.head_dim).transpose(1, 2)

        # Compute attention scores and apply them to the values
        attn = torch.einsum('bhid,bhjd->bhij', q, k)
        attn = attn.softmax(dim=-1)
        x = torch.einsum('bhij,bhjd->bhid', attn, v)

        # enables concat?
        x = x.transpose(1, 2).reshape(b, n, -1)
        # removed projection

        return x

class SimplifiedViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.attention = ShapedAttention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),  # NonLin
            nn.Linear(mlp_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attention(x) + x
        x = self.mlp(x) + x
        # no normalization in paper
        return x