"""Train a model: next-token cross-entropy, teacher forcing, caption dropout, early stopping.

Usage: python -m src.train --config configs/train.yaml
"""
import argparse

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--overfit-one-batch", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    raise NotImplementedError


if __name__ == "__main__":
    main()
