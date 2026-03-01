"""
ChangeFormer — Transformer-based Siamese Change Detection Network
=================================================================

Architecture:
  1. **Encoder**: MiT-b1 (Mix Vision Transformer) backbone from SegFormer
     - Pretrained on ImageNet (via HuggingFace nvidia/mit-b1)
     - Shared weights (Siamese) applied to both before & after images
     - Produces multi-scale feature maps at 4 resolutions
     - Hidden sizes: [64, 128, 320, 512]
  2. **Difference module**: Computes |F_after - F_before| at each scale
  3. **Decoder**: Lightweight MLP decoder fuses multi-scale difference
     features and outputs a 2-class (change / no-change) prediction

Pretrained: MiT-b1 encoder is pretrained on ImageNet-1k.
The decoder can be optionally loaded with LEVIR-CD fine-tuned weights.

Paper: "A Transformer-Based Siamese Network for Change Detection"
License: MIT (open source, free to use)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel
import logging
import os

logger = logging.getLogger(__name__)


# ============================================================
#  MiT ENCODER (SegFormer backbone via HuggingFace)
# ============================================================

class MiTEncoder(nn.Module):
    """
    Mix Vision Transformer encoder (SegFormer backbone).
    Extracts multi-scale features at 4 levels.
    Hidden sizes for mit-b1: [64, 128, 320, 512]
    """

    def __init__(self, model_name: str = "nvidia/mit-b1"):
        super().__init__()
        self.backbone = SegformerModel.from_pretrained(
            model_name, use_safetensors=True
        )
        self.channels = self.backbone.config.hidden_sizes  # [64, 128, 320, 512]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W] normalized image tensor
        Returns:
            list of 4 feature tensors [B, C, H, W] at different scales
        """
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states are already [B, C, H, W] from SegformerModel
        # There are 4 stage outputs
        return list(outputs.hidden_states)


# ============================================================
#  DIFFERENCE MODULE
# ============================================================

class DifferenceModule(nn.Module):
    """Compute absolute difference of Siamese features at each scale."""

    def forward(self, feats_a, feats_b):
        return [torch.abs(fa - fb) for fa, fb in zip(feats_a, feats_b)]


# ============================================================
#  MLP DECODER
# ============================================================

class MLPDecoder(nn.Module):
    """
    Lightweight MLP decoder that fuses multi-scale difference features.
    Each scale is projected to embed_dim, upsampled to 1/4 resolution,
    concatenated, then classified into 2 classes (no-change, change).
    """

    def __init__(self, in_channels: list, embed_dim: int = 256, num_classes: int = 2):
        super().__init__()

        # Linear projection for each scale
        self.linear_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for ch in in_channels
        ])

        # Fusion: 4 scales concatenated -> embed_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, num_classes, 1),
        )

    def forward(self, diff_feats: list, target_size: tuple):
        projected = []
        for i, feat in enumerate(diff_feats):
            proj = self.linear_proj[i](feat)
            proj = F.interpolate(proj, size=target_size, mode="bilinear", align_corners=False)
            projected.append(proj)

        fused = self.fusion(torch.cat(projected, dim=1))
        out = self.classifier(fused)
        return out


# ============================================================
#  CHANGEFORMER FULL MODEL
# ============================================================

class ChangeFormer(nn.Module):
    """
    ChangeFormer: Transformer-based Siamese change detection model.

    Input: Two images (before, after) -- each [B, 3, H, W], normalized
    Output: Change probability map [B, 1, H, W] in range [0, 1]
    """

    def __init__(self, model_name: str = "nvidia/mit-b1", embed_dim: int = 256):
        super().__init__()
        self.encoder = MiTEncoder(model_name)
        self.diff_module = DifferenceModule()
        self.decoder = MLPDecoder(
            in_channels=self.encoder.channels,
            embed_dim=embed_dim,
            num_classes=2,
        )

    def forward(self, before: torch.Tensor, after: torch.Tensor):
        # Siamese encoder (shared weights)
        feats_before = self.encoder(before)
        feats_after = self.encoder(after)

        # Absolute difference at each scale
        diff_feats = self.diff_module(feats_before, feats_after)

        # Decode: target size = 1/4 of input
        h, w = before.shape[2], before.shape[3]
        target_h, target_w = h // 4, w // 4
        logits = self.decoder(diff_feats, (target_h, target_w))

        # Upsample to full resolution
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

        # Softmax: take "change" class probability (class 1)
        probs = F.softmax(logits, dim=1)
        change_prob = probs[:, 1:2, :, :]  # [B, 1, H, W]

        return change_prob

    @torch.no_grad()
    def predict(self, before: torch.Tensor, after: torch.Tensor,
                threshold: float = 0.5):
        """
        Inference helper -- returns binary change mask and probabilities.
        Returns: (binary_mask, change_prob) both [B, 1, H, W]
        """
        self.eval()
        change_prob = self.forward(before, after)
        return (change_prob >= threshold).float(), change_prob


# ============================================================
#  MODEL BUILDER (singleton)
# ============================================================

_model_cache = None


def get_changeformer(weights_path: str = None, device: str = None):
    """
    Build and cache ChangeFormer model.

    Uses ImageNet-pretrained MiT-b1 encoder (Siamese).
    If weights_path provided, loads fine-tuned decoder weights.
    """
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Building ChangeFormer model on {device}...")
    print(f"Loading ChangeFormer (MiT-b1 backbone) on {device}...", flush=True)

    model = ChangeFormer(model_name="nvidia/mit-b1", embed_dim=256)

    if weights_path and os.path.exists(weights_path):
        logger.info(f"Loading fine-tuned weights from {weights_path}")
        state = torch.load(weights_path, map_location=device, weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print("ChangeFormer: loaded fine-tuned weights", flush=True)
    else:
        print("ChangeFormer: using ImageNet-pretrained encoder", flush=True)

    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ChangeFormer ready: {param_count:.1f}M parameters on {device}", flush=True)

    _model_cache = model
    return model
