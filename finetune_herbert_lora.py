import os
import time
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

import csv
from transformers import TrainerCallback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType


def log_metrics_csv(path: Path, row: dict, field_order: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        if new:
            w.writeheader()
        w.writerow(row)


def f1_macro_np(pred: np.ndarray, gold: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (gold == c))
        fp = np.sum((pred == c) & (gold != c))
        fn = np.sum((pred != c) & (gold == c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))


def compute_metrics(eval_pred, num_labels: int):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float((preds == labels).mean())
    f1 = f1_macro_np(preds, labels, num_classes=num_labels)
    return {"accuracy": acc, "f1_macro": f1}


def count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def make_class_weights(train_labels: List[int], num_labels: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.array(train_labels, dtype=np.int64), minlength=num_labels)
    total = counts.sum()
    w = total / (counts + 1e-9)
    w = w * (num_labels / w.sum())  # normalizacja
    return torch.tensor(w, dtype=torch.float32, device=device)


def make_sample_weights(train_labels: List[int], num_labels: int) -> torch.Tensor:
    counts = np.bincount(np.array(train_labels, dtype=np.int64), minlength=num_labels)
    inv = 1.0 / (counts + 1e-9)
    weights = inv[np.array(train_labels, dtype=np.int64)]
    return torch.tensor(weights, dtype=torch.double)  

class SamplerTrainer(Trainer):
    def __init__(self, sample_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: train_dataset is None.")

        num_samples = len(self.sample_weights)

        sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=num_samples,
            replacement=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )



class WeightedTrainer(Trainer):

    def __init__(self, class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


@torch.no_grad()
def measure_inference_time(model, tokenizer, dataset, max_samples: Optional[int] = None, batch_size: int = 64):
    model.eval()
    device = next(model.parameters()).device

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    t0 = time.perf_counter()
    n = 0

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        _ = model(**enc)
        n += len(batch["text"])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    sec = t1 - t0
    return {"inference_s": float(sec), "examples_per_s": float(n / max(sec, 1e-9))}

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, out_dir: Path, params_total: int):
        self.out_dir = Path(out_dir)
        self.params_total = int(params_total)
        self.run_start = time.time()
        self.epoch_start = None

        self.csv_path = self.out_dir / "train_log.csv"
        self.fields = ["epoch", "split", "loss", "acc", "f1", "epoch_time_s", "elapsed_s", "lr", "params"]

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "epoch" not in logs:
            return

        # Trainer daje epoch jako float (np. 1.0), bierzemy int
        epoch = int(round(float(logs["epoch"])))

        is_eval = "eval_loss" in logs
        split = "valid" if is_eval else "train"

        loss = logs.get("eval_loss", logs.get("loss", None))
        acc  = logs.get("eval_accuracy", logs.get("accuracy", None))
        f1   = logs.get("eval_f1_macro", logs.get("f1_macro", None))
        lr   = logs.get("learning_rate", None)

        epoch_time = (time.time() - self.epoch_start) if self.epoch_start else None
        elapsed = time.time() - self.run_start

        row = {
            "epoch": epoch,
            "split": split,
            "loss": float(loss) if loss is not None else "",
            "acc": float(acc) if acc is not None else "",
            "f1": float(f1) if f1 is not None else "",
            "epoch_time_s": float(epoch_time) if epoch_time is not None else "",
            "elapsed_s": float(elapsed),
            "lr": float(lr) if lr is not None else "",
            "params": self.params_total,
        }
        log_metrics_csv(self.csv_path, row, self.fields)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--valid_parquet", required=True)
    ap.add_argument("--test_parquet", required=True)

    ap.add_argument("--model_id", default="allegro/herbert-base-cased")
    ap.add_argument("--output_dir", default="out_herbert_lora")

    ap.add_argument("--num_labels", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=256)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", nargs="+", default=["query", "value"])  

    # trening
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--class_weights", action="store_true")
    ap.add_argument("--weighted_sampler", action="store_true", help="Use WeightedRandomSampler instead of class weights.")

    args = ap.parse_args()
    set_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_files = {
        "train": args.train_parquet,
        "validation": args.valid_parquet,
        "test": args.test_parquet,
    }
    ds = load_dataset("parquet", data_files=data_files)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.model_max_length = args.max_len

    def tokenize_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tok(
            batch["text"],
            truncation=True,
            max_length=args.max_len,
        )

    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in ("text", "label")])
    ds_tok = ds_tok.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tok)

    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=args.num_labels,
    )

    base.resize_token_embeddings(len(tok))

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_w = None
    if args.class_weights and (not args.weighted_sampler):
        train_labels = ds_tok["train"]["labels"]
        class_w = make_class_weights(train_labels, args.num_labels, device)

    sample_w = None
    if args.weighted_sampler:
        train_labels = ds_tok["train"]["labels"]
        sample_w = make_sample_weights(train_labels, args.num_labels)

    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,

        logging_strategy="epoch",

        fp16=args.fp16,
        bf16=args.bf16,

        report_to="none",
        seed=args.seed,
    )

    trainer_kwargs = dict(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, num_labels=args.num_labels),
    )

    if args.weighted_sampler:
        trainer = SamplerTrainer(sample_weights=sample_w, **trainer_kwargs)
    elif args.class_weights:
        trainer = WeightedTrainer(class_weights=class_w, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    params = count_params(model)
    
    params_total = params["total"]
    trainer.add_callback(CSVLoggerCallback(out, params_total=params_total))

    train_start = time.time()
    trainer.train()
    train_end = time.time()

    train_time = train_end - train_start
    (out / "train_time.json").write_text(
        json.dumps({"train_time_s": train_time}, indent=2),
        encoding="utf-8",
    )
    print(f"Train time: {train_time:.1f}s")

    test_metrics = trainer.evaluate(ds_tok["test"])
    print("\n[TEST METRICS]")
    print(test_metrics)

    inf = measure_inference_time(trainer.model, tok, ds["test"], max_samples=None, batch_size=args.eval_batch_size)
    print("\n[INFERENCE]")
    print(inf)

    trainer.model.save_pretrained(out / "lora_adapter")
    tok.save_pretrained(out / "tokenizer")

    (out / "test_metrics.json").write_text(
        json.dumps(
            {
                "test": {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float, np.number))},
                "inference": inf,
                "params_total": params["total"],
                "params_trainable": params["trainable"],
                "train_time_s": train_time,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
