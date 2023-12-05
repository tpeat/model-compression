import torch
from torch import nn
from einops.layers.torch import Rearrange

from models.simplified_block import SimplifiedViTBlock as ViTBlock

# Define the Vision Transformer Model
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, num_classes):
        super(ViT, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = num_patches

        # Image patches embedding
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * 3, dim)
        )

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, dim))

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(*[ViTBlock(dim, heads, mlp_dim) for _ in range(depth)])

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x) + self.positional_encoding
        x = self.transformer_blocks(x)
        x = self.head(x.mean(dim=1))
        return x


