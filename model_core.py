import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np

# ==========================================
# Config Constants
# ==========================================
SEQ_LEN = 11
D_MODEL = 256
DIM_FF = 1024
N_HEADS = 8
N_LAYERS = 3
FOCAL_GAMMA = 2.0
LABEL_SMOOTH = 0.05
DROPOUT = 0.25
INPUT_DIM = 320  # ESM-2 embedding dim (esm2_t6_8M_UR50D)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA2IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_SIZE = len(AMINO_ACIDS)

# Global in-memory cache for ESM embeddings
_ESM_CACHE = {}

def load_esm_cache(esm_cache_dir):
    """Loads the precomputed ESM cache from disk into memory."""
    global _ESM_CACHE
    cache_path = os.path.join(esm_cache_dir, "esm_embeddings.pt")
    if os.path.exists(cache_path) and len(_ESM_CACHE) == 0:
        print(f"[ESM] Loading cached embeddings from {cache_path}")
        try:
            _ESM_CACHE.update(torch.load(cache_path, map_location="cpu"))
            print(f"[ESM] Loaded {len(_ESM_CACHE)} cached embeddings")
        except Exception as e:
            print(f"[WARNING] Failed to load ESM cache: {e}")
    return _ESM_CACHE

def encode_sequence_with_fallback(seq, original_seq=None):
    """Encodes a sequence to ESM embedding.
    For augmented sequences: uses the original sequence's embedding but zero-outs changed positions.
    """
    global _ESM_CACHE
    if seq in _ESM_CACHE:
        return _ESM_CACHE[seq].clone()

    if original_seq and original_seq in _ESM_CACHE:
        enc = _ESM_CACHE[original_seq].clone()
        for i, (a, b) in enumerate(zip(seq, original_seq)):
            if a != b:
                enc[i] = 0.0  # Mask changed/augmented positions
        return enc

    # Fallback to zeros if not found
    return torch.zeros(len(seq), INPUT_DIM, dtype=torch.float32)

# ==========================================
# AASSPE Layer
# ==========================================
class AASSPE(nn.Module):
    def __init__(self, d_model=D_MODEL, seq_len=SEQ_LEN, dropout=DROPOUT):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.d_model = d_model

        center = (seq_len - 1) / 2.0
        distances = torch.tensor(
            [abs(pos - center) for pos in range(seq_len)],
            dtype=torch.float32
        ).unsqueeze(1)

        div_term_even = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        div_term_odd = torch.exp(
            torch.arange(1, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(distances * div_term_even)
        pe[:, 1::2] = torch.cos(distances * div_term_odd)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])

# ==========================================
# BMAM Attention Layer
# ==========================================
class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, key_mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
        if key_mask is not None:
            scores = scores.masked_fill(key_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v), attn

class BMAM(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q_orig = nn.Linear(d_model, d_model)
        self.W_K_orig = nn.Linear(d_model, d_model)
        self.W_V_orig = nn.Linear(d_model, d_model)
        self.W_Q_trans = nn.Linear(SEQ_LEN, SEQ_LEN)
        self.W_K_trans = nn.Linear(SEQ_LEN, SEQ_LEN)
        self.W_V_trans = nn.Linear(SEQ_LEN, SEQ_LEN)

        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.attn = ScaledDotProductAttention()
        
        # Save last attention weights for explainability
        self.last_attn_weights = None

    def _multi_head(self, W_Q, W_K, W_V, x, n_heads, d_head, key_mask=None):
        B, L, D = x.shape
        q = W_Q(x).view(B, L, n_heads, d_head).transpose(1, 2)
        k = W_K(x).view(B, L, n_heads, d_head).transpose(1, 2)
        v = W_V(x).view(B, L, n_heads, d_head).transpose(1, 2)
        ctx, attn_w = self.attn(q, k, v, key_mask=key_mask)
        self.last_attn_weights = attn_w.detach()  # Cache weights in evaluation mode
        return ctx.transpose(1, 2).contiguous().view(B, L, D)

    def forward(self, x, key_mask=None):
        residual = x
        B, L, D = x.shape

        Z = self._multi_head(
            self.W_Q_orig, self.W_K_orig, self.W_V_orig,
            x, self.n_heads, self.d_head, key_mask=key_mask
        )

        x_t = x.transpose(1, 2)
        q_t = self.W_Q_trans(x_t)
        k_t = self.W_K_trans(x_t)
        v_t = self.W_V_trans(x_t)
        scores_t = torch.matmul(q_t, k_t.transpose(-1, -2)) / math.sqrt(SEQ_LEN)
        attn_t = torch.softmax(scores_t, dim=-1)
        Z_t = torch.matmul(attn_t, v_t).transpose(1, 2)

        combined = torch.cat([Z, Z_t], dim=-1)
        out = self.drop(self.proj(combined))
        return self.norm(out + residual)

# ==========================================
# Encoder Layer
# ==========================================
class PoswiseFeedForwardConv(nn.Module):
    def __init__(self, d_model=D_MODEL, dim_ff=1024, dropout=DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, dim_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(dim_ff, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        y = F.relu(self.conv1(x.transpose(1, 2)))
        y = self.conv2(y).transpose(1, 2)
        return self.norm(self.drop(y) + residual)

class EncoderLayer(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, dim_ff=1024, dropout=DROPOUT):
        super().__init__()
        self.bmam = BMAM(d_model, n_heads, dropout)
        self.ffn = PoswiseFeedForwardConv(d_model, dim_ff, dropout)

    def forward(self, x, key_mask=None):
        x = self.bmam(x, key_mask=key_mask)
        x = self.ffn(x)
        return x

# ==========================================
# Main DeepTESite Model
# ==========================================
class DeepTESite(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(input_dim, d_model)  # 320 -> 256
        self.pos = AASSPE(d_model, seq_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, dim_ff=max(2048, dim_ff), dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Primary head (Methylation)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_features(self, x):
        key_mask_1d = (x.abs().sum(dim=-1) == 0)
        key_mask = key_mask_1d.unsqueeze(1).unsqueeze(2)
        x = self.proj(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, key_mask=key_mask)
        return x[:, self.seq_len // 2, :]  # Extract center residue embedding

    def forward(self, x):
        features = self.get_features(x)
        return self.head(features)

# Alias to center on user's model name
HybridArgMe = DeepTESite

# ==========================================
# Custom Loss and Scheduler
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTH):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_cls = logits.size(1)
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        with torch.no_grad():
            smooth = torch.full_like(prob, self.smoothing / (n_cls - 1))
            smooth.scatter_(1, targets.view(-1, 1), 1.0 - self.smoothing)
        p_t = prob.gather(1, targets.view(-1, 1)).squeeze(1)
        lp_t = (smooth * log_prob).sum(dim=1)
        return (-(1 - p_t) ** self.gamma * lp_t).mean()

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup = warmup_steps
        self.total = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup:
            lr = self.base_lr * self._step / self.warmup
        else:
            prog = (self._step - self.warmup) / max(1, self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                       1 + math.cos(math.pi * prog))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

# ==========================================
# Peptide Datasets
# ==========================================
class PeptideDataset(Dataset):
    """Uses ESM cache for encoding with fallback for augmented sequences."""
    def __init__(self, samples_with_originals):
        self.X = [encode_sequence_with_fallback(seq, orig)
                  for seq, _, orig in samples_with_originals]
        self.y = [label for _, label, _ in samples_with_originals]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], torch.tensor(self.y[idx], dtype=torch.long)

class PeptideDatasetSimple(Dataset):
    """For test/val data without augmentation (no original_seq needed)."""
    def __init__(self, samples):
        self.X = [encode_sequence_with_fallback(seq, seq) for seq, _ in samples]
        self.y = [label for _, label in samples]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], torch.tensor(self.y[idx], dtype=torch.long)

def make_weighted_sampler(dataset):
    labels = dataset.y
    counts = np.bincount(labels)
    weights = 1.0 / counts
    sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float32)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
