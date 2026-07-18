"""Deferred GPU training entry point. Do not run without optional PyTorch dependencies."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from src.data.dataset import ManifestDataset
from src.model.omr import OMRTransformer
from src.vocab.tokenizer import Tokenizer


class TorchOMRDataset(Dataset):
    def __init__(self, manifest: Path, split: str, image_size: tuple[int, int], max_curriculum: int):
        self.base = ManifestDataset(manifest, split, max_curriculum)
        self.image_size = image_size
        self.tokenizer = Tokenizer()

    def __len__(self): return len(self.base)

    def __getitem__(self, index):
        image, target, _ = self.base[index]
        image = ImageOps.pad(image, self.image_size, color=255)
        return 1.0 - TF.to_tensor(image), torch.tensor(self.tokenizer.encode(target))


def collate(batch, pad_id):
    images, sequences = zip(*batch)
    return torch.stack(images), nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_id)


def train(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    tokenizer = Tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dataset = TorchOMRDataset(args.manifest, "train", (args.width, args.height), args.max_curriculum)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=lambda b: collate(b, tokenizer.pad_id), pin_memory=device.type == "cuda")
    model = OMRTransformer(len(tokenizer.vocabulary), tokenizer.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for images, ids in loader:
            images, ids = images.to(device), ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images, ids[:, :-1])
                loss = criterion(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); running += loss.item()
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "config": vars(args), "vocabulary": tokenizer.vocabulary}
        torch.save(checkpoint, args.output / f"epoch-{epoch:03d}.pt")
        print(json.dumps({"epoch": epoch, "train_loss": running / max(1, len(loader)), "device": str(device)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024); parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--max-curriculum", type=int, default=2); parser.add_argument("--seed", type=int, default=1729)
    train(parser.parse_args())


if __name__ == "__main__": main()
