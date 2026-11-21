"""Train a model: next-token cross-entropy, teacher forcing, caption dropout, early stopping.

Usage: python -m src.train --config configs/train.yaml
       python -m src.train --conditioning none            # override train.yaml's conditioning
       python -m src.train --overfit-one-batch             # wiring check: loss should go near 0

Logs to Weights & Biases when train.yaml has `log: wandb` (run `wandb login` once first):
config, data sizes and parameter count at the start; train loss, lr and grad norm per step;
val loss and perplexity per epoch. The best checkpoint (lowest val loss) goes to
<checkpoint_dir>/<run_name>/best.pt with everything needed to rebuild the model.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import PhraseDataset, collate, load_captions
from src.data.phrases import load_phrases
from src.model.solo_model import SoloModel
from src.tokens import remi

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> tuple[dict, dict]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    with open(ROOT / cfg["model_config"]) as f:
        model_cfg = yaml.safe_load(f)
    model_cfg["dropout"] = cfg.get("dropout", model_cfg.get("dropout", 0.1))  # train.yaml wins
    return cfg, model_cfg


def data_version() -> str:
    """Short hash of the split files, so runs on different data are told apart."""
    h = hashlib.sha1()
    for name in ("train", "val", "test"):
        p = ROOT / "data" / "splits" / f"{name}.txt"
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()[:8]


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min((step - warmup) / max(1, total - warmup), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return f


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def lm_loss(model: SoloModel, batch: dict, loss_fn) -> torch.Tensor:
    """Teacher forcing: tokens 0..L-2 in, tokens 1..L-1 as targets."""
    inp = {**batch, **{k: batch[k][:, :-1] for k in ("ids", "bar_pos", "beat")}}
    target = batch["ids"][:, 1:]
    logits = model(inp)
    return loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))


@torch.no_grad()
def evaluate(model: SoloModel, loader: DataLoader, device) -> float:
    """Mean per-token cross-entropy (no label smoothing) over the loader."""
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=remi.pad_id(), reduction="sum")
    total, tokens = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        total += lm_loss(model, batch, loss_fn).item()
        tokens += batch["ids"][:, 1:].ne(remi.pad_id()).sum().item()
    model.train()
    return total / max(tokens, 1)


class Logger:
    def __init__(self, cfg: dict, model_cfg: dict, run_name: str, enabled: bool):
        self.run = None
        if enabled:
            import wandb
            self.run = wandb.init(project=cfg.get("wandb_project", "jazz-solo-gen"),
                                  entity=cfg.get("wandb_entity"), name=run_name,
                                  group=cfg["conditioning"],
                                  config={"train": cfg, "model": model_cfg})

    def config(self, **kv):
        if self.run:
            self.run.config.update(kv, allow_val_change=True)

    def log(self, metrics: dict, step: int):
        if self.run:
            self.run.log(metrics, step=step)

    def summary(self, **kv):
        if self.run:
            self.run.summary.update(kv)

    def finish(self):
        if self.run:
            self.run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--overfit-one-batch", action="store_true")
    ap.add_argument("--conditioning", choices=["none", "chords", "full"],
                    help="override train.yaml's conditioning")
    ap.add_argument("--max-epochs", type=int, help="override train.yaml's max_epochs")
    ap.add_argument("--no-wandb", action="store_true", help="don't log, whatever train.yaml says")
    args = ap.parse_args()
    cfg, model_cfg = load_config(args.config)
    if args.conditioning:
        cfg["conditioning"] = args.conditioning
    if args.max_epochs:
        cfg["max_epochs"] = args.max_epochs
    cond = cfg["conditioning"]

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data
    with open(ROOT / "configs" / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    captions = None
    if cond == "full":
        cap_path = ROOT / data_cfg["paths"]["captions"]
        if not cap_path.exists():
            raise FileNotFoundError(f"conditioning=full needs captions at {cap_path} (H2)")
        captions = load_captions(cap_path)
    max_len = model_cfg["max_len"] + 1  # +1: the model sees L-1 tokens after the shift
    train_set = PhraseDataset(load_phrases("train"), max_len,
                              transpose=(data_cfg["transpose"]["min"], data_cfg["transpose"]["max"]),
                              captions=captions, caption_dropout=cfg["caption_dropout"],
                              seed=cfg["seed"])
    val_set = PhraseDataset(load_phrases("val"), max_len, captions=captions, random_caption=False)
    bs = cfg["batch_size"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train_set, bs, shuffle=True, collate_fn=collate, generator=gen)
    val_loader = DataLoader(val_set, bs, shuffle=False, collate_fn=collate)
    print(f"train {len(train_set)} phrases ({train_set.n_dropped} over {max_len} tokens dropped), "
          f"val {len(val_set)} ({val_set.n_dropped} dropped), device {device}")

    # ---- model, optimizer
    model = SoloModel(model_cfg, cond).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = 1 if args.overfit_one_batch else len(train_loader)
    total_steps = (cfg.get("overfit_steps", 500) if args.overfit_one_batch
                   else cfg["max_epochs"] * steps_per_epoch)
    warmup = 0 if args.overfit_one_batch else cfg["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(max(warmup, 1), total_steps))
    smoothing = 0.0 if args.overfit_one_batch else cfg["label_smoothing"]
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=remi.pad_id(), label_smoothing=smoothing)
    clip = cfg.get("grad_clip", 1.0)

    run_name = f"{cond}-{Path(cfg['model_config']).stem}-{time.strftime('%m%d-%H%M')}"
    if args.overfit_one_batch:
        run_name = "overfit-" + run_name
    use_wandb = cfg.get("log") == "wandb" and not args.no_wandb
    logger = Logger(cfg, model_cfg, run_name, use_wandb)
    logger.config(n_params=n_params, n_train=len(train_set), n_val=len(val_set),
                  data_version=data_version(), device=str(device), run_name=run_name)
    print(f"{run_name}: {n_params / 1e6:.2f}M trainable parameters")

    # ---- overfit one batch
    if args.overfit_one_batch:
        batch = to_device(next(iter(train_loader)), device)
        for step in range(total_steps):
            loss = lm_loss(model, batch, loss_fn)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, clip)
            opt.step()
            sched.step()
            logger.log({"train/loss": loss.item()}, step)
            if step % 50 == 0 or step == total_steps - 1:
                print(f"step {step:4d}  loss {loss.item():.4f}")
        logger.summary(final_loss=loss.item())
        logger.finish()
        return

    # ---- train
    ckpt_dir = ROOT / cfg["checkpoint_dir"] / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best, bad_epochs, step = float("inf"), 0, 0
    for epoch in range(cfg["max_epochs"]):
        t0, running = time.time(), 0.0
        for i, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            loss = lm_loss(model, batch, loss_fn)
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, clip).item()
            opt.step()
            sched.step()
            running += loss.item()
            logger.log({"train/loss": loss.item(), "train/lr": sched.get_last_lr()[0],
                        "train/grad_norm": gnorm, "epoch": epoch}, step)
            step += 1
            if i % 100 == 0:
                print(f"epoch {epoch} step {i}/{steps_per_epoch}  loss {loss.item():.4f}")

        val = evaluate(model, val_loader, device)
        train_avg = running / steps_per_epoch
        logger.log({"val/loss": val, "val/ppl": math.exp(val), "train/epoch_loss": train_avg,
                    "epoch": epoch, "epoch_time_s": time.time() - t0}, step)
        print(f"epoch {epoch}: train {train_avg:.4f}  val {val:.4f}  ppl {math.exp(val):.2f}  "
              f"({time.time() - t0:.0f}s)")

        if val < best:
            best, bad_epochs = val, 0
            torch.save({"model": model.state_dict(), "model_cfg": model_cfg, "train_cfg": cfg,
                        "conditioning": cond, "epoch": epoch, "step": step, "val_loss": val,
                        "data_version": data_version()}, ckpt_dir / "best.pt")
            logger.summary(best_val_loss=val, best_val_ppl=math.exp(val), best_epoch=epoch)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stop_patience"]:
                print(f"early stop: no val improvement for {bad_epochs} epochs")
                break

    print(f"best val loss {best:.4f} (ppl {math.exp(best):.2f}) -> {ckpt_dir / 'best.pt'}")
    logger.finish()


if __name__ == "__main__":
    main()
