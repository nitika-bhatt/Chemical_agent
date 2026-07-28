import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, matthews_corrcoef,
                             roc_auc_score, f1_score, confusion_matrix)

# Import model components from model_core
from model_core import (
    DeepTESite, FocalLoss, WarmupCosineScheduler,
    PeptideDataset, PeptideDatasetSimple, make_weighted_sampler,
    SEQ_LEN, INPUT_DIM, AMINO_ACIDS,
    encode_sequence_with_fallback, _ESM_CACHE
)

# ==========================================
# Config / Environment Check
# ==========================================
IS_KAGGLE = os.path.exists("/kaggle/working")
IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")

if IS_KAGGLE:
    BASE_DIR       = "/kaggle/input/datasets/kartikj31/protein-data-11/data-11"
    CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    SPLIT_DIR      = "/kaggle/working/splits"
    ESM_CACHE_DIR  = "/kaggle/working/esm_cache"
    NUM_WORKERS    = 2
    PIN_MEMORY     = True
    print("[INFO] Kaggle environment detected")
elif IS_COLAB:
    # Dynamic Colab path resolution
    if os.path.exists("/content/drive/MyDrive/ProteinMind_Agent/data-11"):
        BASE_DIR = "/content/drive/MyDrive/ProteinMind_Agent/data-11"
    elif os.path.exists("data-11"):
        BASE_DIR = "data-11"
    elif os.path.exists("/content/drive/MyDrive/protein_data/data-11"):
        BASE_DIR = "/content/drive/MyDrive/protein_data/data-11"
    else:
        BASE_DIR = "/content/drive/MyDrive/ProteinMind_Agent/data-11"
    
    CHECKPOINT_DIR = "checkpoints"
    SPLIT_DIR      = "splits"
    ESM_CACHE_DIR  = "esm_cache"
    NUM_WORKERS    = 2
    PIN_MEMORY     = True
    print(f"[INFO] Google Colab environment detected. Using dataset from: {BASE_DIR}")
else:
    # Local fallback
    _base = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR       = os.path.join(_base, "data-11") if os.path.exists(os.path.join(_base, "data-11")) else "data-11"
    CHECKPOINT_DIR = "checkpoints"
    SPLIT_DIR      = "splits"
    ESM_CACHE_DIR  = "esm_cache"
    NUM_WORKERS    = 0
    PIN_MEMORY     = False
    print(f"[INFO] Local environment detected. Using dataset from: {BASE_DIR}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(ESM_CACHE_DIR, exist_ok=True)

# Training Hyperparameters
MASK_RATE      = 2
AUG_ROUNDS     = 3
D_MODEL        = 256
DIM_FF         = 1024
N_HEADS        = 8
N_LAYERS       = 3
FOCAL_GAMMA    = 2.0
LABEL_SMOOTH   = 0.05
LR             = 2e-4
WEIGHT_DECAY   = 5e-3
BATCH_SIZE     = 128 if (IS_KAGGLE or IS_COLAB) else 64
ACCUM_STEPS    = 2
EPOCHS         = 120
DROPOUT        = 0.25
WARMUP_STEPS   = 300
PATIENCE       = 30
SWA_START_FRAC = 0.8
SWA_LR         = 1e-4
MIXUP_ALPHA    = 0.23
N_FOLDS        = 5
SEED           = 42
TEST_SIZE      = 0.20

BALANCED_TEST_N = 260
TRAIN_NEG_RATIO = 1

# ESM-2 Settings
ESM_MODEL_NAME = "esm2_t6_8M_UR50D"
ESM_BATCH_SIZE = 256

ALL_SPECIES = ["Human", "Chimpanzee", "Mouse", "Rat"]
MOD_TYPES   = ["Asymmetric_dimethylarginine", "Dimethylated_arginine",
               "Omega-N-methylarginine", "Symmetric_dimethylarginine"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# ==========================================
# ESM-2 Embedding Precomputation
# ==========================================
def _load_esm_model():
    """Load ESM-2 model and alphabet. Cached after first call."""
    print(f"[ESM] Loading {ESM_MODEL_NAME} ...")
    model, alphabet = torch.hub.load("facebookresearch/esm", ESM_MODEL_NAME)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    print(f"[ESM] Model loaded. Embed dim={INPUT_DIM}")
    return model, alphabet, batch_converter

def precompute_esm_embeddings(sequences):
    """Precompute ESM-2 per-residue embeddings for all unique sequences."""
    global _ESM_CACHE
    cache_path = os.path.join(ESM_CACHE_DIR, "esm_embeddings.pt")
    if os.path.exists(cache_path) and len(_ESM_CACHE) == 0:
        print(f"[ESM] Loading cached embeddings from {cache_path}")
        _ESM_CACHE.update(torch.load(cache_path, map_location="cpu"))
        print(f"[ESM] Loaded {len(_ESM_CACHE)} cached embeddings")

    unique_seqs = list(set(sequences))
    missing = [s for s in unique_seqs if s not in _ESM_CACHE]

    if not missing:
        print(f"[ESM] All {len(unique_seqs)} sequences already cached ✓")
        return

    print(f"[ESM] Need to compute {len(missing)} new embeddings (have {len(_ESM_CACHE)} cached)")
    esm_model, alphabet, batch_converter = _load_esm_model()

    for i in range(0, len(missing), ESM_BATCH_SIZE):
        batch_seqs = missing[i:i+ESM_BATCH_SIZE]
        data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[6])

        embeddings = results["representations"][6]

        for j, seq in enumerate(batch_seqs):
            emb = embeddings[j, 1:len(seq)+1, :].cpu()
            _ESM_CACHE[seq] = emb

        if (i // ESM_BATCH_SIZE) % 10 == 0:
            print(f"    [ESM] Processed {min(i+ESM_BATCH_SIZE, len(missing))}/{len(missing)}")

    del esm_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    torch.save(_ESM_CACHE, cache_path)
    print(f"[ESM] Saved {len(_ESM_CACHE)} embeddings to {cache_path}")

# Import local encoder helpers
import model_core
model_core._ESM_CACHE = _ESM_CACHE

# ==========================================
# Split & Save Dataset Utilities
# ==========================================
def _read_sequences_from_csv(fpath):
    df = pd.read_csv(fpath)
    if "sequence" in df.columns:
        seq_col = "sequence"
    elif "Sequence" in df.columns:
        seq_col = "Sequence"
    else:
        seq_col = df.columns[-1]
    seqs = df[seq_col].astype(str).str.strip().str.upper()
    seqs = seqs[seqs.str.len() == SEQ_LEN].tolist()
    return seqs

def split_and_save_data(base_dir=BASE_DIR, split_dir=SPLIT_DIR, test_size=TEST_SIZE, seed=SEED):
    all_done = all(
        os.path.exists(os.path.join(split_dir, f"{sp.lower()}_trainval.csv"))
        and os.path.exists(os.path.join(split_dir, f"{sp.lower()}_test_raw.csv"))
        and os.path.exists(os.path.join(split_dir, f"{sp.lower()}_test_bal.csv"))
        for sp in ALL_SPECIES
    )
    if all_done and os.path.exists(os.path.join(split_dir, "combined_test_bal.csv")):
        print("[SPLIT] Split files already exist — skipping.\n")
        return

    print("[SPLIT] Creating splits per species ...")
    rng = random.Random(seed)
    per_species_bal_rows = {}

    for species in ALL_SPECIES:
        pos_seqs, neg_seqs = [], []
        sp_path = os.path.join(base_dir, species)
        if not os.path.exists(sp_path):
            print(f"  [WARNING] Species path not found: {sp_path}")
            continue
        for mod_type in MOD_TYPES:
            mod_path = os.path.join(sp_path, mod_type)
            if not os.path.isdir(mod_path):
                continue
            for fname in os.listdir(mod_path):
                if not fname.endswith(".csv"):
                    continue
                fpath = os.path.join(mod_path, fname)
                seqs = _read_sequences_from_csv(fpath)
                if "positive" in fname.lower():
                    pos_seqs.extend(seqs)
                elif "negative" in fname.lower():
                    neg_seqs.extend(seqs)

        pos_seqs = list(set(pos_seqs))
        neg_seqs = list(set(neg_seqs))
        print(f"  [{species}] pos={len(pos_seqs)}  neg={len(neg_seqs)}")

        if not pos_seqs or not neg_seqs:
            print(f"  [WARNING] Skipping {species} — empty pos or neg set")
            continue

        pos_train, pos_test = train_test_split(pos_seqs, test_size=test_size, random_state=seed)
        neg_train, neg_test = train_test_split(neg_seqs, test_size=test_size, random_state=seed)

        trainval_rows = [(s, 1) for s in pos_train] + [(s, 0) for s in neg_train]
        rng.shuffle(trainval_rows)
        pd.DataFrame(trainval_rows, columns=["sequence", "label"]).to_csv(
            os.path.join(split_dir, f"{species.lower()}_trainval.csv"), index=False)

        test_raw_rows = [(s, 1) for s in pos_test] + [(s, 0) for s in neg_test]
        pd.DataFrame(test_raw_rows, columns=["sequence", "label"]).to_csv(
            os.path.join(split_dir, f"{species.lower()}_test_raw.csv"), index=False)

        n_bal = min(len(pos_test), len(neg_test))
        if BALANCED_TEST_N is not None:
            n_bal = min(n_bal, BALANCED_TEST_N)
        bal_pos = rng.sample(pos_test, n_bal)
        bal_neg = rng.sample(neg_test, n_bal)
        test_bal_rows = [(s, 1) for s in bal_pos] + [(s, 0) for s in bal_neg]
        rng.shuffle(test_bal_rows)
        pd.DataFrame(test_bal_rows, columns=["sequence", "label"]).to_csv(
            os.path.join(split_dir, f"{species.lower()}_test_bal.csv"), index=False)

        print(f"    trainval={len(trainval_rows)}  "
              f"test_raw={len(test_raw_rows)} (pos={len(pos_test)}, neg={len(neg_test)})  "
              f"test_bal={len(test_bal_rows)} (1:1) ✓")
        per_species_bal_rows[species] = test_bal_rows

    comb_rows = []
    for species, rows in per_species_bal_rows.items():
        comb_rows.extend(rows)
    rng.shuffle(comb_rows)
    pd.DataFrame(comb_rows, columns=["sequence", "label"]).to_csv(
        os.path.join(split_dir, "combined_test_bal.csv"), index=False)

    n_pos_comb = sum(l for _, l in comb_rows)
    n_neg_comb = sum(1 - l for _, l in comb_rows)
    print(f"\n  [COMBINED BALANCED TEST] {len(comb_rows)} samples  (pos={n_pos_comb}, neg={n_neg_comb}) ✓")
    print("[SPLIT] Done.\n")

def load_trainval_samples(split_dir=SPLIT_DIR):
    all_samples = []
    for species in ALL_SPECIES:
        path = os.path.join(split_dir, f"{species.lower()}_trainval.csv")
        if not os.path.exists(path):
            print(f"[WARNING] trainval not found: {path}")
            continue
        df = pd.read_csv(path)
        samples = list(zip(df["sequence"].tolist(), df["label"].tolist()))
        print(f"  [{species}] trainval: {len(samples)}")
        all_samples.extend(samples)
    print(f"  [COMBINED] Total trainval: {len(all_samples)}")
    return all_samples

def load_csv_samples(path):
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return list(zip(df["sequence"].tolist(), df["label"].tolist()))

# ==========================================
# Data Balance & Augmentations
# ==========================================
def balance_samples(samples, neg_ratio=TRAIN_NEG_RATIO, seed=SEED):
    pos = [(s, l) for s, l in samples if l == 1]
    neg = [(s, l) for s, l in samples if l == 0]
    print(f"    [BALANCE] Before -> Pos: {len(pos)}  Neg: {len(neg)}")
    rng = random.Random(seed)
    target_neg = min(len(neg), len(pos) * neg_ratio)
    neg = rng.sample(neg, target_neg)
    if neg_ratio == 1 and len(pos) > len(neg):
        pos = rng.sample(pos, len(neg))
    balanced = pos + neg
    rng.shuffle(balanced)
    print(f"    [BALANCE] After  -> Pos: {len(pos)}  Neg: {len(neg)}  Total: {len(balanced)}")
    return balanced

def augment_samples(samples, mask_rate=MASK_RATE, rounds=AUG_ROUNDS):
    augmented = [(seq, label, seq) for seq, label in samples]
    for r in range(rounds):
        for seq, label in samples:
            chars = list(seq)
            center = len(chars) // 2
            positions = [i for i in range(len(chars)) if i != center]
            n = min(mask_rate, len(positions))
            chosen = random.sample(positions, n)
            masked = chars.copy()
            for idx in chosen:
                masked[idx] = "X"
            augmented.append(("".join(masked), label, seq))
            if r % 2 == 0:
                chosen2 = random.sample(positions, n)
                subst = chars.copy()
                for idx in chosen2:
                    new_aa = random.choice(AMINO_ACIDS)
                    while new_aa == subst[idx]:
                        new_aa = random.choice(AMINO_ACIDS)
                    subst[idx] = new_aa
                augmented.append(("".join(subst), label, seq))
    return augmented

# ==========================================
# Mixup Helpers
# ==========================================
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

# ==========================================
# Training & Evaluation Loops
# ==========================================
def train_epoch(model, loader, optimizer, scheduler, criterion, use_mixup=True, accum_steps=ACCUM_STEPS):
    model.train()
    tot_loss = tot_correct = tot_n = 0
    optimizer.zero_grad()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        if use_mixup:
            mixed_x, y_a, y_b, lam = mixup_data(x, y)
            out = model(mixed_x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam) / accum_steps
        else:
            out = model(x)
            loss = criterion(out, y) / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        tot_loss    += loss.item() * accum_steps * len(y)
        tot_correct += (out.argmax(1) == y).sum().item()
        tot_n       += len(y)
    return tot_loss / tot_n, tot_correct / tot_n

@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    preds, labels, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        pred = (prob >= threshold).astype(int)
        probs.extend(prob)
        preds.extend(pred)
        labels.extend(y.numpy())
    
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    f1  = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    cm  = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sn = tp / (tp + fn + 1e-9)
        sp = tn / (tn + fp + 1e-9)
    else:
        sn = sp = 0.0
    return dict(acc=acc, mcc=mcc, f1=f1, auc=auc, sn=sn, sp=sp, preds=preds, labels=labels, probs=probs)

@torch.no_grad()
def evaluate_ensemble(models, loader, threshold=0.5):
    for m in models:
        m.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        avg_prob = torch.zeros(x.size(0), device=x.device)
        for m in models:
            out = m(x)
            prob = torch.softmax(out, dim=1)[:, 1]
            avg_prob += prob
        avg_prob /= len(models)
        all_probs.extend(avg_prob.cpu().numpy())
        all_labels.extend(y.numpy())
    
    probs_arr = np.array(all_probs)
    preds = (probs_arr >= threshold).astype(int)
    acc = accuracy_score(all_labels, preds)
    mcc = matthews_corrcoef(all_labels, preds)
    f1  = f1_score(all_labels, preds, zero_division=0)
    auc = roc_auc_score(all_labels, probs_arr) if len(set(all_labels)) > 1 else float("nan")
    cm  = confusion_matrix(all_labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sn = tp / (tp + fn + 1e-9)
        sp = tn / (tn + fp + 1e-9)
    else:
        sn = sp = 0.0
    return dict(acc=acc, mcc=mcc, f1=f1, auc=auc, sn=sn, sp=sp, preds=preds, labels=all_labels, probs=all_probs)

def find_best_threshold(probs, labels):
    best_t, best_mcc = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        p = (np.array(probs) >= t).astype(int)
        m = matthews_corrcoef(labels, p)
        if m > best_mcc:
            best_mcc = m
            best_t   = t
    return best_t, best_mcc

# ==========================================
# Train Fold Script
# ==========================================
def train_fold(train_samples_aug, val_samples, batch_size, epochs, patience, lr=LR, warmup_steps=WARMUP_STEPS):
    train_ds = PeptideDataset(train_samples_aug)
    val_ds   = PeptideDatasetSimple(val_samples)
    sampler  = make_weighted_sampler(train_ds)
    train_ld = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model     = DeepTESite().to(device)
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, len(train_ld) // ACCUM_STEPS) * epochs
    warmup      = min(warmup_steps, total_steps // 5)
    scheduler   = WarmupCosineScheduler(optimizer, warmup, total_steps)

    best_mcc, best_state = -1.0, None
    no_improve           = 0
    swa_start_ep         = max(1, int(epochs * SWA_START_FRAC))

    from torch.optim.swa_utils import AveragedModel
    swa_model  = AveragedModel(model)
    swa_active = False

    for ep in range(1, epochs + 1):
        use_mixup = ep < swa_start_ep
        tr_loss, tr_acc = train_epoch(model, train_ld, optimizer, scheduler, criterion, use_mixup)
        if ep >= swa_start_ep:
            swa_model.update_parameters(model)
            if not swa_active:
                print(f"      [SWA] Activated at epoch {ep}")
                swa_active = True

        m   = evaluate(model, val_ld)
        tag = " *" if m["mcc"] > best_mcc else ""
        if ep % 10 == 0 or m["mcc"] > best_mcc:
            print(f"      Ep {ep:3d}  TrLoss={tr_loss:.4f}  TrAcc={tr_acc:.4f}  Acc={m['acc']:.4f}  MCC={m['mcc']:.4f}  F1={m['f1']:.4f}{tag}")
        if m["mcc"] > best_mcc:
            best_mcc   = m["mcc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"      [EARLY STOP] epoch {ep}")
            break

    if swa_active:
        try:
            torch.optim.swa_utils.update_bn(train_ld, swa_model, device=device)
            m_swa = evaluate(swa_model, val_ld)
            if m_swa["mcc"] > best_mcc:
                print(f"      [SWA] Improved MCC: {best_mcc:.4f} → {m_swa['mcc']:.4f}")
                best_mcc   = m_swa["mcc"]
                best_state = {k: v.cpu().clone() for k, v in swa_model.module.state_dict().items()}
            else:
                print(f"      [SWA] No improvement (SWA={m_swa['mcc']:.4f} vs best={best_mcc:.4f})")
        except Exception as e:
            print(f"      [SWA] BN update failed ({e}), using best checkpoint")

    model.load_state_dict(best_state)
    model.to(device)
    m      = evaluate(model, val_ld)
    best_t, _ = find_best_threshold(m["probs"], m["labels"])
    m      = evaluate(model, val_ld, threshold=best_t)
    return best_state, m, best_t

# ==========================================
# Ensemble Test Evaluation
# ==========================================
def evaluate_on_test(models, threshold=0.5):
    print("\n" + "=" * 70)
    print(f"  FINAL TEST SET EVALUATION (ENSEMBLE of {len(models)} folds)")
    print(f"  Threshold: {threshold:.3f}")
    print("=" * 70)

    per_sp_bal = {}
    for species in ALL_SPECIES:
        path = os.path.join(SPLIT_DIR, f"{species.lower()}_test_bal.csv")
        samples = load_csv_samples(path)
        if not samples:
            continue
        ds = PeptideDatasetSimple(samples)
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        m = evaluate_ensemble(models, ld, threshold=threshold)
        n_pos = sum(l for _, l in samples)
        n_neg = sum(1 - l for _, l in samples)
        print(f"\n  [{species}] Balanced (n={len(samples)}, pos={n_pos}, neg={n_neg})")
        print(f"    ACC : {m['acc']:.4f}   MCC : {m['mcc']:.4f}   F1 : {m['f1']:.4f}   AUC : {m['auc']:.4f}")
        per_sp_bal[species] = m

    comb_bal_path = os.path.join(SPLIT_DIR, "combined_test_bal.csv")
    comb_bal = load_csv_samples(comb_bal_path)
    if comb_bal:
        ds_cb = PeptideDatasetSimple(comb_bal)
        ld_cb = DataLoader(ds_cb, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        m_cb = evaluate_ensemble(models, ld_cb, threshold=threshold)
        print(f"\n  [COMBINED BALANCED] (n={len(comb_bal)})")
        print(f"    ACC : {m_cb['acc']:.4f}   MCC : {m_cb['mcc']:.4f}   AUC : {m_cb['auc']:.4f}")
        per_sp_bal["Combined_Balanced"] = m_cb

    per_sp_raw = {}
    for species in ALL_SPECIES:
        path = os.path.join(SPLIT_DIR, f"{species.lower()}_test_raw.csv")
        samples = load_csv_samples(path)
        if not samples:
            continue
        ds = PeptideDatasetSimple(samples)
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        m = evaluate_ensemble(models, ld, threshold=threshold)
        per_sp_raw[species] = m

    rows = []
    for sp_name, m in per_sp_bal.items():
        rows.append(dict(split="balanced", species=sp_name, acc=round(m["acc"], 4), mcc=round(m["mcc"], 4), f1=round(m["f1"], 4), sn=round(m["sn"], 4), specificity=round(m["sp"], 4), auc=round(m["auc"], 4)))
    for sp_name, m in per_sp_raw.items():
        rows.append(dict(split="raw_imbalanced", species=sp_name, acc=round(m["acc"], 4), mcc=round(m["mcc"], 4), f1=round(m["f1"], 4), sn=round(m["sn"], 4), specificity=round(m["sp"], 4), auc=round(m["auc"], 4)))

    save_path = os.path.join(CHECKPOINT_DIR, "test_results_esm_ensemble.csv")
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"\n  [SAVED] {save_path}")
    return per_sp_bal, per_sp_raw

# ==========================================
# Main Execution Trigger
# ==========================================
def main():
    print("=" * 70)
    print("  DeepTESite — Training & CV Pipeline (Ensemble Mode)")
    print("=" * 70)

    # Step 0: Create data splits
    split_and_save_data()

    # Step 1: Load trainval data
    raw = load_trainval_samples(SPLIT_DIR)
    if not raw:
        print("[FATAL] No trainval data found! Please verify d:/mtechthesis/New folder/data-11 is populated.")
        return

    balanced = balance_samples(raw, neg_ratio=TRAIN_NEG_RATIO)

    # Step 1.5: Precompute ESM embeddings for all unique sequences
    all_seqs = [s for s, _ in balanced]
    for species in ALL_SPECIES:
        for suffix in ["_test_bal.csv", "_test_raw.csv"]:
            path = os.path.join(SPLIT_DIR, f"{species.lower()}{suffix}")
            samples = load_csv_samples(path)
            all_seqs.extend([s for s, _ in samples])
    comb_path = os.path.join(SPLIT_DIR, "combined_test_bal.csv")
    comb_samples = load_csv_samples(comb_path)
    all_seqs.extend([s for s, _ in comb_samples])

    print(f"\n[ESM] Precomputing embeddings for {len(set(all_seqs))} unique sequences...")
    precompute_esm_embeddings(all_seqs)

    seqs = [s for s, _ in balanced]
    labels = [l for _, l in balanced]

    # Step 2: 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_states, fold_thresholds, fold_metrics = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(seqs, labels)):
        print(f"\n  -- Fold {fold+1}/{N_FOLDS} --")
        train_samples = [(seqs[i], labels[i]) for i in train_idx]
        val_samples   = [(seqs[i], labels[i]) for i in val_idx]
        train_aug     = augment_samples(train_samples, MASK_RATE, AUG_ROUNDS)
        print(f"    Train: {len(train_samples)} → {len(train_aug)} (aug) | Val: {len(val_samples)}")

        state, metrics, threshold = train_fold(train_aug, val_samples, BATCH_SIZE, EPOCHS, PATIENCE)
        fold_states.append(state)
        fold_thresholds.append(threshold)
        fold_metrics.append(metrics)
        print(f"    Fold {fold+1} Completed | MCC: {metrics['mcc']:.4f} | Thr: {threshold:.2f}")

    # CV Summary
    avg_acc = np.mean([m["acc"] for m in fold_metrics])
    avg_mcc = np.mean([m["mcc"] for m in fold_metrics])
    avg_f1  = np.mean([m["f1"]  for m in fold_metrics])
    avg_auc = np.mean([m["auc"] for m in fold_metrics])
    avg_thr = np.mean(fold_thresholds)

    print(f"\n  -- CV Results ({N_FOLDS}-Fold) --")
    print(f"    Avg ACC : {avg_acc:.4f} | Avg MCC : {avg_mcc:.4f} | Avg F1 : {avg_f1:.4f} | Avg AUC : {avg_auc:.4f} | Avg Thr : {avg_thr:.3f}")

    # Save Checkpoint File
    save_path = os.path.join(CHECKPOINT_DIR, "pretrained_esm_v4.pt")
    torch.save({
        "fold_states": fold_states,
        "fold_thresholds": fold_thresholds,
        "avg_threshold": avg_thr,
        "config": dict(input_dim=INPUT_DIM, d_model=D_MODEL, n_heads=N_HEADS,
                       n_layers=N_LAYERS, dim_ff=DIM_FF, seq_len=SEQ_LEN,
                       encoding="ESM-2", esm_model=ESM_MODEL_NAME),
    }, save_path)
    print(f"    [SAVED Checkpoint] {save_path}")

    # Ensemble test evaluation
    ensemble_models = []
    for state in fold_states:
        m = DeepTESite().to(device)
        m.load_state_dict(state)
        m.eval()
        ensemble_models.append(m)

    evaluate_on_test(ensemble_models, threshold=avg_thr)

    # ── Step 3: Train Consensus Meta-Learner ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  Step 3: Training ConsensusAgent Meta-Learner (Accuracy Booster)")
    print("=" * 70)
    try:
        from agents.train_consensus import train_consensus_meta_learner
        consensus_ckpt = os.path.join(CHECKPOINT_DIR, "consensus_agent_meta_learner.pkl")
        train_consensus_meta_learner(
            checkpoint_path=save_path,
            data_dir=BASE_DIR,
            output_path=consensus_ckpt,
        )
        print(f"  [SAVED Consensus Meta-Learner] {consensus_ckpt}")
    except Exception as e:
        print(f"  [WARN] Consensus meta-learner training skipped: {e}")
        print("  You can train it manually with: python agents/train_consensus.py")

    print("\n[DONE] Full pipeline complete!")

if __name__ == "__main__":
    main()
