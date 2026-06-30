# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic model listing from the timm library."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cached model list (populated on first call)
_models_cache: list[dict[str, Any]] | None = None

# Popular model prefixes — matched against model name before the '.' dataset suffix.
# timm names look like "resnet50.a1_in1k", "vit_base_patch16_224.augreg_in1k", etc.
_POPULAR_PREFIXES = [
    "resnet50",
    "resnet101",
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_b4",
    "vit_base_patch16_224",
    "vit_small_patch16_224",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "mobilenetv3_large_100",
    "eva02_base_patch14_224",
]


def _extract_family(name: str) -> str:
    """Extract architecture family from a timm model name."""
    prefixes = [
        "resnet",
        "resnext",
        "resnetv2",
        "wide_resnet",
        "vit",
        "deit",
        "beit",
        "eva",
        "eva02",
        "efficientnet",
        "efficientnetv2",
        "tf_efficientnet",
        "convnext",
        "convnextv2",
        "swin",
        "swinv2",
        "mobilenetv3",
        "mobilenetv2",
        "regnet",
        "regnety",
        "regnetx",
        "densenet",
        "inception",
        "maxvit",
        "coatnet",
        "nfnet",
        "dm_nfnet",
    ]
    name_lower = name.lower()
    for prefix in sorted(prefixes, key=len, reverse=True):
        if name_lower.startswith(prefix):
            return prefix
    # Fallback: first segment before underscore or digit
    parts = name.split("_")
    return parts[0] if parts else name


def list_models(pretrained_only: bool = True) -> list[dict[str, Any]]:
    """List available timm models, grouped by family.

    Returns:
        List of dicts with name, family, popular, pretrained fields.

    """
    global _models_cache

    if _models_cache is not None:
        return _models_cache

    try:
        import timm

        names = timm.list_models(pretrained=pretrained_only)
    except ImportError:
        logger.warning("timm not installed — model list will be empty")
        _models_cache = []
        return _models_cache
    except Exception:
        logger.exception("Failed to list timm models")
        _models_cache = []
        return _models_cache

    popular_set = set(_POPULAR_PREFIXES)
    models: list[dict[str, Any]] = []
    for name in sorted(names):
        # Extract base name (before dataset suffix): "resnet50.a1_in1k" → "resnet50"
        base = name.split(".")[0]
        entry: dict[str, Any] = {
            "name": name,
            "family": _extract_family(name),
            "popular": base in popular_set,
            "pretrained": True,
        }

        # Enrich with pretrained config metadata (fast dict lookup, no model instantiation)
        try:
            cfg = timm.get_pretrained_cfg(name)
            if cfg is not None:
                input_size = getattr(cfg, "input_size", None)
                if input_size:
                    entry["image_size"] = input_size[-1]
                entry["num_classes"] = getattr(cfg, "num_classes", None)
                tag = getattr(cfg, "tag", "") or ""
                if "in21k" in tag:
                    entry["dataset"] = "ImageNet-21k"
                elif "in22k" in tag:
                    entry["dataset"] = "ImageNet-22k"
                elif "in1k" in tag:
                    entry["dataset"] = "ImageNet-1k"
                elif tag:
                    entry["dataset"] = tag
        except Exception:
            pass

        models.append(entry)

    _models_cache = models
    logger.info("Cached %d timm models", len(models))
    return _models_cache


def get_model_info(model_name: str) -> dict[str, Any]:
    """Get details about a specific timm model.

    Args:
        model_name: The timm model identifier.

    Returns:
        Dict with input_size, num_features, num_params, mean, std.

    """
    try:
        import timm

        model = timm.create_model(model_name, pretrained=False)
        cfg: dict[str, Any] = model.default_cfg if hasattr(model, "default_cfg") else {}
        input_size = cfg.get("input_size", (3, 224, 224))
        num_features = getattr(model, "num_features", 0)
        num_params = sum(p.numel() for p in model.parameters())

        del model  # Free memory immediately

        return {
            "name": model_name,
            "input_size": list(input_size),
            "image_size": input_size[-1] if input_size else 224,
            "num_features": num_features,
            "num_params": num_params,
            "num_params_m": round(num_params / 1e6, 1),
            "mean": list(cfg.get("mean", (0.485, 0.456, 0.406))),
            "std": list(cfg.get("std", (0.229, 0.224, 0.225))),
        }
    except Exception as e:
        return {"name": model_name, "error": str(e)}
