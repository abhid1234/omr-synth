"""Compact image-to-token encoder/decoder. Requires the optional training dependencies."""
from __future__ import annotations

import math

import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, d_model: int = 384, max_patches: int = 4096) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, d_model, 4, stride=4),
        )
        self.position = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, 6, 1536, 0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, 6, norm=nn.LayerNorm(d_model))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images).flatten(2).transpose(1, 2)
        if features.size(1) > self.position.size(1):
            raise ValueError("image creates more patches than max_patches")
        return self.transformer(features + self.position[:, :features.size(1)])


class OMRTransformer(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, d_model: int = 384,
                 layers: int = 6, max_target: int = 1024) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.encoder = ImageEncoder(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.target_position = nn.Parameter(torch.randn(1, max_target, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model, 6, 1536, 0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(d_model))
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(images)
        length = input_ids.size(1)
        target = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        target = target + self.target_position[:, :length]
        causal = nn.Transformer.generate_square_subsequent_mask(length, device=input_ids.device)
        decoded = self.decoder(target, memory, tgt_mask=causal,
                               tgt_key_padding_mask=input_ids.eq(self.pad_id))
        return self.output(decoded)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, bos_id: int, eos_id: int, max_length: int = 512) -> torch.Tensor:
        ids = torch.full((images.size(0), 1), bos_id, device=images.device, dtype=torch.long)
        for _ in range(max_length - 1):
            next_id = self(images, ids)[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat((ids, next_id), dim=1)
            if torch.all(next_id.squeeze(1).eq(eos_id)):
                break
        return ids
