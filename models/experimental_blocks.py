# simplified attention subblock
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShapedAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

        # trainable scalars
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.alpha * torch.eye(n, device=dots.device) + self.beta * F.softmax(dots, dim=-1) - self.gamma * 1/n

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class SASBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = ShapedAttention(dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    

class SASVPBlock(SASBlock):
    def __init__(self, dim, heads):
        super().__init__(dim, heads)
        self.beta_v = nn.Parameter(torch.full((dim,), 0.1))
        self.beta_p = nn.Parameter(torch.full((dim,), 0.1))

        self.wv_init = nn.Linear(dim, dim, bias=False)
        self.wp_init = nn.Linear(dim, dim, bias=False)

        # Initialize to identity
        nn.init.eye_(self.wv_init.weight)
        nn.init.eye_(self.wp_init.weight)

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = self.beta_v * self.wv_init(x) + self.beta_p * self.wp_init(x)
        x = residual + x
        x = x + self.mlp(self.ln2(x))
        return x
    

class SASPBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = ShapedAttention(dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.beta_ff = nn.Parameter(torch.full((1,), 0.1))
        self.beta_sa = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, x):
        x_norm = self.ln(x)
        x_mha = self.beta_sa * self.attn(x_norm)
        x_mlp = self.beta_ff * self.mlp(x_norm)
        return x + x_mha + x_mlp