import os
import math
import time
import json
import csv
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from datasets import Dataset as HFDataset
from transformers import AutoTokenizer
from tqdm import tqdm


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

SCALER = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
AMP_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

def accuracy(pred: torch.Tensor, gold: torch.Tensor) -> float:
    return (pred == gold).float().mean().item()

def f1_macro(pred: torch.Tensor, gold: torch.Tensor, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (gold == c)).sum().item()
        fp = ((pred == c) & (gold != c)).sum().item()
        fn = ((pred != c) & (gold == c)).sum().item()
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(sum(f1s) / num_classes)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def log_metrics_csv(path: Path, row: dict, field_order: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        if new:
            w.writeheader()
        w.writerow(row)

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ParquetTextClsDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, max_len: int):
        self.ds = HFDataset.from_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[idx]
        text = ex["text"]
        label = int(ex["label"])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            add_special_tokens=True,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def collate_cls(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    maxL = max(x["input_ids"].numel() for x in batch)
    B = len(batch)
    input_ids = torch.full((B, maxL), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, maxL), dtype=torch.long)
    labels = torch.stack([b["labels"] for b in batch])

    for i, b in enumerate(batch):
        L = b["input_ids"].numel()
        input_ids[i, :L] = b["input_ids"]
        attention_mask[i, :L] = b["attention_mask"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return self.drop(x + self.pe[:L].unsqueeze(0))

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)  # [B,h,L,dk]

    def _merge(self, x):
        B, h, L, dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, h * dk)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self._split(self.q(x))
        k = self._split(self.k(x))
        v = self._split(self.v(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # [B,h,L,L]
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        p = torch.softmax(scores, dim=-1)
        p = self.drop(p)
        y = torch.matmul(p, v)  # [B,h,L,dk]
        return self.o(self._merge(y))

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(F.gelu(self.w1(x))))

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadedAttention(n_heads, d_model, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model, eps=1e-5)
        self.n2 = nn.LayerNorm(d_model, eps=1e-5)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.n1(x), attn_mask=attn_mask))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x

class DecoderOnlyBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        pad_id: int,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout, max_len=max_len)
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)[None, None, :, :]  # [1,1,L,L]

    def _pad_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [B,L] -> [B,1,1,L] True at PAD positions
        return (input_ids == self.pad_id)[:, None, None, :]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        h = self.tok(input_ids) * math.sqrt(self.tok.embedding_dim)
        h = self.pos(h)

        cm = self._causal_mask(L, input_ids.device)     # [1,1,L,L]
        pm = self._pad_mask(input_ids)                  # [B,1,1,L]
        attn_mask = cm | pm                             # broadcast -> [B,1,L,L]

        for blk in self.layers:
            h = blk(h, attn_mask)

        return self.norm(h)  # [B,L,D]

class DecoderOnlyClassifier(nn.Module):
    def __init__(self, backbone: DecoderOnlyBackbone, d_model: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.backbone(input_ids)  # [B,L,D]

        m = attention_mask.unsqueeze(-1).to(h.dtype)  # [B,L,1]
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # [B,D]

        return self.head(self.drop(pooled))  # [B,C]

def make_class_weights(train_parquet: str, num_labels: int, device) -> Optional[torch.Tensor]:
    ds = HFDataset.from_parquet(train_parquet)
    labels = ds["label"]
    counts = [0] * num_labels
    for y in labels:
        counts[int(y)] += 1
    total = sum(counts)
    w = [total / (c + 1e-9) for c in counts]
    s = sum(w)
    w = [x * (num_labels / s) for x in w]
    return torch.tensor(w, dtype=torch.float32, device=device)


def make_sampler(train_parquet: str, num_labels: int) -> WeightedRandomSampler:
    ds = HFDataset.from_parquet(train_parquet)
    labels = list(map(int, ds["label"]))

    counts = [0] * num_labels
    for y in labels:
        counts[y] += 1

    class_w = [1.0 / (c + 1e-12) for c in counts]
    sample_w = [class_w[y] for y in labels]

    return WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )




def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt,
    num_labels: int,
    class_weights: Optional[torch.Tensor],
    accum_steps: int,
    max_grad_norm: float,
) -> Dict[str, float]:
    model.train()
    device = next(model.parameters()).device
    opt.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_n = 0
    all_pred, all_gold = [], []

    pbar = tqdm(loader, desc="train", dynamic_ncols=True)
    for step, batch in enumerate(pbar, 1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            logits = model(input_ids, attn)
            loss = F.cross_entropy(logits, y, weight=class_weights) / accum_steps

        SCALER.scale(loss).backward()

        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            SCALER.step(opt)
            SCALER.update()
            opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            total_loss += loss.item() * accum_steps * y.size(0)
            total_n += y.size(0)
            pred = logits.argmax(dim=-1)
            all_pred.append(pred.detach().cpu())
            all_gold.append(y.detach().cpu())

        if step % 20 == 0:
            pbar.set_postfix_str(f"loss~{(total_loss/max(1,total_n)):.4f}")

    pred = torch.cat(all_pred)
    gold = torch.cat(all_gold)
    avg_loss = total_loss / max(1, total_n)
    acc = accuracy(pred, gold)
    f1 = f1_macro(pred, gold, num_labels)

    return {"loss": avg_loss, "acc": acc, "f1": f1}

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    num_labels: int,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_n = 0
    all_pred, all_gold = [], []

    for batch in tqdm(loader, desc="eval", leave=False, dynamic_ncols=True):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        with torch.cuda.amp.autocast(dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            logits = model(input_ids, attn)
            loss = F.cross_entropy(logits, y, weight=class_weights)

        total_loss += loss.item() * y.size(0)
        total_n += y.size(0)
        pred = logits.argmax(dim=-1)
        all_pred.append(pred.detach().cpu())
        all_gold.append(y.detach().cpu())

    pred = torch.cat(all_pred)
    gold = torch.cat(all_gold)
    avg_loss = total_loss / max(1, total_n)
    acc = accuracy(pred, gold)
    f1 = f1_macro(pred, gold, num_labels)
    return {"loss": avg_loss, "acc": acc, "f1": f1}

@torch.no_grad()
def measure_inference_time(model: nn.Module, loader: DataLoader, warmup_batches: int = 5) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device

    # warmup
    it = iter(loader)
    for _ in range(warmup_batches):
        try:
            b = next(it)
        except StopIteration:
            break
        _ = model(b["input_ids"].to(device), b["attention_mask"].to(device))

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n = 0
    for b in loader:
        _ = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        n += b["input_ids"].size(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    sec = t1 - t0
    ex_s = n / max(sec, 1e-9)
    return {"inference_s": sec, "examples_per_s": ex_s}

def lr_schedule(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (0.1 + 0.9 * cosine)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--valid_parquet", required=True)
    ap.add_argument("--test_parquet", required=True)

    ap.add_argument("--output_dir", default="out_from_scratch_cls")
    ap.add_argument("--tokenizer_id", default="allegro/herbert-base-cased",
                    help="Tokenizer can be pre-trained; model weights are trained from scratch.")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--num_labels", type=int, default=3)

    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max_pos", type=int, default=512)

    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weights", action="store_true", help="Use inverse-frequency class weights.")
    ap.add_argument("--balanced_sampler", action="store_true",
                help="Use WeightedRandomSampler (oversampling minority classes) in training.")

    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    ds_train = ParquetTextClsDataset(args.train_parquet, tokenizer, args.max_len)
    ds_valid = ParquetTextClsDataset(args.valid_parquet, tokenizer, args.max_len)
    ds_test  = ParquetTextClsDataset(args.test_parquet, tokenizer, args.max_len)

    
    train_sampler = None
    if args.balanced_sampler:
        train_sampler = make_sampler(args.train_parquet, args.num_labels)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda b: collate_cls(b, pad_id),
    )


    dl_valid = DataLoader(
        ds_valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        pin_memory=True,
        collate_fn=lambda b: collate_cls(b, pad_id),
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        pin_memory=True,
        collate_fn=lambda b: collate_cls(b, pad_id),
    )

    backbone = DecoderOnlyBackbone(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_pos,
        pad_id=pad_id,
    )
    model = DecoderOnlyClassifier(backbone, d_model=args.d_model, num_labels=args.num_labels, dropout=args.dropout).to(device)

    if hasattr(tokenizer, "__len__"):
        full_vocab = len(tokenizer)
        if full_vocab != vocab_size:
            # resize embedding
            old = model.backbone.tok
            new_emb = nn.Embedding(full_vocab, old.embedding_dim, padding_idx=pad_id).to(device)
            with torch.no_grad():
                new_emb.weight[: old.num_embeddings].copy_(old.weight)
            model.backbone.tok = new_emb
            vocab_size = full_vocab

    params = count_params(model)
    (out / "run_info.json").write_text(
        json.dumps(
            {
                "tokenizer_id": args.tokenizer_id,
                "vocab_size": vocab_size,
                "pad_id": pad_id,
                "num_labels": args.num_labels,
                "model_cfg": {
                    "d_model": args.d_model,
                    "n_layers": args.n_layers,
                    "n_heads": args.n_heads,
                    "d_ff": args.d_ff,
                    "dropout": args.dropout,
                    "max_pos": args.max_pos,
                },
                "params": params,
                "max_len": args.max_len,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Model params: {params:,d}")

    class_w = make_class_weights(args.train_parquet, args.num_labels, device) if args.class_weights else None
    if class_w is not None:
        print("Using class weights:", class_w.detach().cpu().tolist())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    log_path = out / "train_log.csv"
    log_fields = [
        "epoch", "split",
        "loss", "acc", "f1",
        "epoch_time_s", "elapsed_s",
        "lr", "params",
    ]

    steps_per_epoch = len(dl_train)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    best_f1 = -1.0
    global_step = 0
    run_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        for pg in opt.param_groups:
            pg["lr"] = lr_schedule(global_step, total_steps, warmup_steps, args.lr)

        tr = train_one_epoch(model, dl_train, opt, args.num_labels, class_w, args.accum_steps, args.max_grad_norm)

        global_step += steps_per_epoch

        cur_lr = opt.param_groups[0]["lr"]

        va = evaluate(model, dl_valid, args.num_labels, class_w)

        epoch_sec = time.time() - epoch_start
        elapsed_sec = time.time() - run_start

        print(
            f"[epoch {epoch}] "
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} f1={tr['f1']:.4f} | "
            f"valid loss={va['loss']:.4f} acc={va['acc']:.4f} f1={va['f1']:.4f} | "
            f"lr={cur_lr:.2e} | epoch={epoch_sec:.1f}s"
        )

        log_metrics_csv(log_path, {
            "epoch": epoch, "split": "train",
            "loss": round(tr["loss"], 6),
            "acc": round(tr["acc"], 6),
            "f1": round(tr["f1"], 6),
            "epoch_time_s": round(epoch_sec, 3),
            "elapsed_s": round(elapsed_sec, 3),
            "lr": f"{cur_lr:.3e}",
            "params": params,
        }, log_fields)

        log_metrics_csv(log_path, {
            "epoch": epoch, "split": "valid",
            "loss": round(va["loss"], 6),
            "acc": round(va["acc"], 6),
            "f1": round(va["f1"], 6),
            "epoch_time_s": round(epoch_sec, 3),
            "elapsed_s": round(elapsed_sec, 3),
            "lr": f"{cur_lr:.3e}",
            "params": params,
        }, log_fields)

        torch.save({"model": model.state_dict(), "args": vars(args)}, out / "last.pt")
        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, out / "best.pt")

    best_ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"], strict=True)

    te = evaluate(model, dl_test, args.num_labels, class_w)
    inf = measure_inference_time(model, dl_test)

    (out / "test_metrics.json").write_text(
        json.dumps(
            {
                "test_loss": te["loss"],
                "test_acc": te["acc"],
                "test_f1_macro": te["f1"],
                "inference_s": inf["inference_s"],
                "examples_per_s": inf["examples_per_s"],
                "params": params,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n[TEST]")
    print(f"loss={te['loss']:.4f} acc={te['acc']:.4f} f1={te['f1']:.4f}")
    print(f"inference: {inf['inference_s']:.3f}s total, {inf['examples_per_s']:.1f} ex/s")
    print(f"params: {params:,d}")

if __name__ == "__main__":
    main()
