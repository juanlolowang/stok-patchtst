"""
model.py
Implementasi PatchTST (Patch Time-Series Transformer) dengan PyTorch.

Referensi:
  Nie et al. (2023) — "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
  https://arxiv.org/abs/2211.14730
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_patches, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# PatchTST
# ---------------------------------------------------------------------------
class PatchTST(nn.Module):
    """
    Patch Time Series Transformer untuk univariate forecasting.

    Args:
        seq_len    : panjang input sekuens (lookback)
        pred_len   : panjang output prediksi (horizon)
        patch_len  : ukuran setiap patch
        stride     : stride antar patch  (biasanya patch_len // 2)
        d_model    : dimensi embedding transformer
        n_heads    : jumlah attention head
        n_layers   : jumlah transformer encoder layer
        d_ff       : dimensi feed-forward layer
        dropout    : dropout rate
        n_channels : jumlah channel / variabel (1 untuk univariate)
    """

    def __init__(
        self,
        seq_len:    int = 6,
        pred_len:   int = 3,
        patch_len:  int = 3,
        stride:     int = 1,
        d_model:    int = 64,
        n_heads:    int = 4,
        n_layers:   int = 2,
        d_ff:       int = 128,
        dropout:    float = 0.1,
        n_channels: int = 1,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.patch_len = patch_len
        self.stride    = stride
        self.n_channels = n_channels

        # Hitung jumlah patch
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Patch Embedding: linear projection dari R^patch_len → R^d_model
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Positional Encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=self.num_patches + 4)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN (lebih stabil)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction Head (flatten → linear ke pred_len)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),           # (B*C, num_patches*d_model)
            nn.Linear(self.num_patches * d_model, pred_len),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)  →  out: (B, pred_len)
        B = batch size, C = channels (1), L = seq_len
        """
        B, C, L = x.shape

        # --- Patching ---
        # Unfold: (B, C, num_patches, patch_len)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # (B*C, num_patches, patch_len)
        patches = patches.reshape(B * C, self.num_patches, self.patch_len)

        # --- Patch Embedding ---
        out = self.patch_embedding(patches)       # (B*C, num_patches, d_model)

        # --- Positional Encoding ---
        out = self.pos_enc(out)

        # --- Transformer ---
        out = self.transformer(out)               # (B*C, num_patches, d_model)

        # --- Head ---
        out = self.head(out)                      # (B*C, pred_len)
        out = out.reshape(B, C, self.pred_len)    # (B, C, pred_len)
        out = out.mean(dim=1)                     # (B, pred_len)  average over channels

        return out


# ---------------------------------------------------------------------------
# Vanilla LSTM (Baseline)
# ---------------------------------------------------------------------------
class VanillaLSTM(nn.Module):
    """
    Baseline LSTM untuk univariate forecasting.
    """
    def __init__(
        self,
        seq_len:    int = 6,
        pred_len:   int = 3,
        hidden_dim: int = 64,
        n_layers:   int = 2,
        dropout:    float = 0.1,
        n_channels: int = 1,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.n_channels = n_channels

        # LSTM Layer
        # input_size=1 karena univariate
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Output Head
        self.head = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)  →  (B, L, C) untuk LSTM
        """
        B, C, L = x.shape
        x = x.permute(0, 2, 1)  # (B, L, C)

        # lstm_out: (B, L, hidden_dim)
        lstm_out, _ = self.lstm(x)

        # Ambil hidden state terakhir: (B, hidden_dim)
        last_hidden = lstm_out[:, -1, :]

        # out: (B, pred_len)
        out = self.head(last_hidden)
        return out
if __name__ == "__main__":
    model = PatchTST(seq_len=6, pred_len=3, patch_len=3, stride=1,
                     d_model=64, n_heads=4, n_layers=2, d_ff=128)
    dummy = torch.randn(32, 1, 6)   # batch=32, channel=1, seq=6
    out   = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")     # (32, 3)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {params:,}")
