from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )


class SepConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


@dataclass(frozen=True)
class TinyDSCNNConfig:
    in_channels: int = 1
    width1: int = 24
    width2: int = 48
    width3: int = 96
    dropout: float = 0.1


class TinyDSCNN(nn.Module):
    def __init__(self, cfg: TinyDSCNNConfig):
        super().__init__()
        self.stem = ConvBNAct(cfg.in_channels, cfg.width1, k=3, s=1, p=1)
        self.block1 = SepConvBNAct(cfg.width1, cfg.width2, stride=2)  # 28 -> 14
        self.block2 = SepConvBNAct(cfg.width2, cfg.width3, stride=2)  # 14 -> 7
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(cfg.width3, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x

    def forward_with_activations(
        self, x: torch.Tensor, *, detach_between: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        acts: list[torch.Tensor] = []
        x = self.stem(x)
        acts.append(x)
        if detach_between:
            x = x.detach()
        x = self.block1(x)
        acts.append(x)
        if detach_between:
            x = x.detach()
        x = self.block2(x)
        acts.append(x)
        if detach_between:
            x = x.detach()
        x = self.head(x)
        acts.append(x)
        return x, acts


@dataclass(frozen=True)
class TinyCNNConfig:
    in_channels: int = 1
    width1: int = 16
    width2: int = 32
    width3: int = 64
    dropout: float = 0.1
    extra_3x3: bool = False  # adds a final 3x3 conv at 7x7


class TinyCNN(nn.Module):
    def __init__(self, cfg: TinyCNNConfig):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(cfg.in_channels, cfg.width1, k=3, s=1, p=1),
            ConvBNAct(cfg.width1, cfg.width2, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2),  # 28 -> 14
            ConvBNAct(cfg.width2, cfg.width3, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2),  # 14 -> 7
            ConvBNAct(cfg.width3, cfg.width3, k=3, s=1, p=1)
            if cfg.extra_3x3
            else nn.Identity(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(cfg.width3, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x

    def forward_with_activations(
        self, x: torch.Tensor, *, detach_between: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        acts: list[torch.Tensor] = []
        x = self.features[0](x)
        acts.append(x)
        if detach_between:
            x = x.detach()
        x = self.features[1](x)
        acts.append(x)
        if detach_between:
            x = x.detach()
        x = self.features[2](x)
        x = self.features[3](x)
        acts.append(x)
        if detach_between:
            x = x.detach()
        x = self.features[4](x)
        x = self.features[5](x)
        if not isinstance(self.features[5], nn.Identity):
            acts.append(x)
            if detach_between:
                x = x.detach()
        x = self.head(x)
        acts.append(x)
        return x, acts


def build_model(cfg: dict[str, Any]) -> nn.Module:
    name = str(cfg.get("name", "tiny_dscnn"))
    if name == "tiny_dscnn":
        model_cfg = TinyDSCNNConfig(
            in_channels=int(cfg.get("in_channels", 1)),
            width1=int(cfg.get("width1", 24)),
            width2=int(cfg.get("width2", 48)),
            width3=int(cfg.get("width3", 96)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
        return TinyDSCNN(model_cfg)
    if name == "tiny_cnn":
        model_cfg = TinyCNNConfig(
            in_channels=int(cfg.get("in_channels", 1)),
            width1=int(cfg.get("width1", 16)),
            width2=int(cfg.get("width2", 32)),
            width3=int(cfg.get("width3", 64)),
            dropout=float(cfg.get("dropout", 0.1)),
            extra_3x3=bool(cfg.get("extra_3x3", False)),
        )
        return TinyCNN(model_cfg)
    raise ValueError(f"Unknown model: {name}")
