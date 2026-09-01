import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DropPath (stochastic depth) ---
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

# --- Sinusoidal PE (batch-first) ---
class PositionalEncodingBF(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, L, d)
    def forward(self, x):  # (B, L, d)
        return x + self.pe[:, :x.size(1), :]

# --- Custom Encoder layer with separate attn/ffn dropout + Pre-LN ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512,
                 attn_dropout=0.1, ffn_dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(ffn_dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pre-LN Transformer
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop_path1(h)
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.drop_path2(h)
        return x

# --- Attention pooling over the sequence ---
class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, 1)
    def forward(self, h):  # (B, L, d)
        w = self.q(h).squeeze(-1)         # (B, L)
        a = torch.softmax(w, dim=-1)      # (B, L)
        return torch.einsum('bl, bld -> bd', a, h)

class TransformerModel(nn.Module):
    def __init__(self, n_features, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=512, attn_dropout=0.1, ffn_dropout=0.1,
                 drop_path_rate=0.1, output_size=1, max_seq_len=1024,
                 pool_type="attn", pool_last_k=64):
        super().__init__()
        self.horizon = output_size
        self.d_model = d_model
        self.pool_type = pool_type
        self.pool_last_k = pool_last_k

        self.input_embedding = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncodingBF(d_model, max_seq_len)

        # Stochastic depth schedule (0 -> drop_path_rate across depth)
        dpr = torch.linspace(0, drop_path_rate, steps=num_layers).tolist()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward,
                         attn_dropout=attn_dropout, ffn_dropout=ffn_dropout,
                         drop_path=dpr[i])
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        if pool_type == "attn":
            self.pool = AttnPool(d_model)
        else:
            self.pool = None  # will mean/avg pool

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_size)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # (B, L, n_features)
        x = self.input_embedding(x)                 # (B, L, d)
        x = x * (self.d_model ** 0.5)               # embed scaling
        x = self.pos_encoding(x)                    # (B, L, d)
        for lyr in self.layers:
            x = lyr(x)
        h = self.norm(x)

        # Readout: attention pool or mean over last K tokens
        if self.pool_type == "attn":
            h_read = self.pool(h)                   # (B, d)
        else:
            k = min(self.pool_last_k, h.size(1))
            h_read = h[:, -k:, :].mean(dim=1)       # (B, d)

        y = self.head(h_read)                       # (B, horizon)
        return y
