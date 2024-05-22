import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT3D(nn.Module):
    def __init__(self, *, 
                 image_size, 
                 image_patch_size, 
                 frames, 
                 frame_patch_size, 
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 pool = 'cls', 
                 channels = 1, 
                 dim_head = 64, 
                 dropout = 0., 
                 emb_dropout = 0.,
                 embedding_dim = 0,
                 pos_mode = "none"):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        self.patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.unpatchify = nn.Sequential(
            Rearrange('b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size, h = image_height // patch_height, w = image_width // patch_width, f = frames // frame_patch_size),
        )

        self.n_pos = 1
        if embedding_dim > 0:
            self.embedding_to_token = nn.Linear(embedding_dim, dim)
            self.n_pos = 2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + self.n_pos, dim))
        self.max_patches = num_patches + self.n_pos
        self.pos_mode = pos_mode

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.num_features = dim

    def add_pos(self, tokens):
        num_tokens = tokens.shape[1]
        pos = self.pos_embedding[:, :self.max_patches]

        if self.pos_mode == "none":
            return tokens + pos[:, :num_tokens] 
        
        elif self.pos_mode == "interpolate":
            pos_cls, pos = pos[:, 0].unsqueeze(1), pos[:, 1:] # [b, 1, d], [b, max_patches - 1, d]

            pos = rearrange(pos, 'b n d -> b d n') # batch, dim, max_patches - 1
            pos = F.interpolate(pos, size = num_tokens - 1, mode = 'linear')
            pos = rearrange(pos, 'b d n -> b n d')

            pos = torch.cat((pos_cls, pos), dim = 1)
            return tokens + pos
        
    def forward(self, video, embedding = None, normalizer = None):
        if normalizer is not None:
            img = normalizer.normalize(img)
        
        if len(video.shape) == 4:
            video = video.unsqueeze(1)  # add channel dimension [b c nt nx ny]

        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        if embedding is not None:
            embedding = self.embedding_to_token(embedding)
            embedding = repeat(embedding, 'b d -> b 1 d')
            x = torch.cat((x, embedding), dim=1)

        x = self.add_pos(x)

        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        return x