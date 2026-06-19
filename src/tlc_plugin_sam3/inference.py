# =============================================================================
# <copyright>
# Copyright (c) 2026 3LC Inc. All rights reserved.
#
# All rights are reserved. Reproduction or transmission in whole or in part, in
# any form or by any means, electronic, mechanical or otherwise, is prohibited
# without the prior written permission of the copyright owner.
# </copyright>
# =============================================================================
"""SAM3 model inference wrapper.

Handles model loading, single-image prediction, and embedding extraction.
Supports multi-class prediction by running one text prompt per class.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as f_nn

logger = logging.getLogger(__name__)

# Singleton model cache
_model = None
_processor = None
_device: str = "cpu"


def _ensure_model(device: str = "cuda") -> tuple[Any, Any]:
    """Load SAM3 model on first call, reuse thereafter."""
    global _model, _processor, _device
    if _processor is not None and _device == device:
        return _model, _processor

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    logger.info("Loading SAM3 model on %s...", device)
    import os

    bpe_path = os.path.join(os.path.dirname(__file__), "bpe_simple_vocab_16e6.txt.gz")
    _model = build_sam3_image_model(bpe_path=bpe_path)
    _model.to(device)
    _model.eval()
    _processor = Sam3Processor(_model, device=device, confidence_threshold=0.2)
    _device = device
    logger.info("SAM3 model loaded.")
    return _model, _processor


def predict_single_image(
    image: Image.Image,
    labels: list[str],
    confidence: float = 0.2,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run SAM3 on a single image for all labels.

    Args:
        image: PIL Image (RGB).
        labels: List of text prompts, e.g. ["fish", "turtle"].
        confidence: Score threshold for detections.
        device: Torch device.

    Returns:
        Dict with keys:
            masks: list of (H, W) uint8 arrays per instance
            boxes: list of [x0, y0, x1, y1] per instance
            scores: list of float per instance
            labels: list of int (class index) per instance
            label_names: list of str (class name) per instance

    """
    _, processor = _ensure_model(device)
    processor.confidence_threshold = confidence

    all_masks = []
    all_boxes = []
    all_scores = []
    all_labels = []
    all_label_names = []

    with torch.no_grad():
        state = processor.set_image(image)

        for label_idx, label_text in enumerate(labels):
            # Reset prompts between classes
            processor.reset_all_prompts(state)
            output = processor.set_text_prompt(prompt=label_text, state=state)

            masks = output.get("masks")
            boxes = output.get("boxes")
            scores = output.get("scores")

            if masks is None or scores is None:
                continue

            masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
            boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
            scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

            # Normalize mask shape to (N, H, W)
            if len(masks_np.shape) == 4:
                masks_np = masks_np[:, 0, :, :]

            for i in range(len(scores_np)):
                all_masks.append(masks_np[i].astype(np.uint8))
                all_boxes.append(boxes_np[i].tolist())
                all_scores.append(float(scores_np[i]))
                all_labels.append(label_idx)
                all_label_names.append(label_text)

    return {
        "masks": all_masks,
        "boxes": all_boxes,
        "scores": all_scores,
        "labels": all_labels,
        "label_names": all_label_names,
    }


def predict_single_image_with_state(
    image: Image.Image,
    labels: list[str],
    confidence: float = 0.2,
    device: str = "cuda",
) -> tuple[dict[str, Any], Any]:
    """Like predict_single_image but also returns inference_state for embeddings.

    Returns:
        (predictions_dict, inference_state)

    """
    _, processor = _ensure_model(device)
    processor.confidence_threshold = confidence

    all_masks = []
    all_boxes = []
    all_scores = []
    all_labels_idx = []
    all_masks_tensor = []  # Keep tensor versions for embedding extraction

    with torch.no_grad():
        state = processor.set_image(image)

        for label_idx, label_text in enumerate(labels):
            processor.reset_all_prompts(state)
            output = processor.set_text_prompt(prompt=label_text, state=state)

            masks = output.get("masks")
            boxes = output.get("boxes")
            scores = output.get("scores")

            if masks is None or scores is None:
                continue

            masks_t = masks if torch.is_tensor(masks) else torch.from_numpy(np.asarray(masks))
            masks_np = masks_t.detach().cpu().numpy()
            boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
            scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

            if len(masks_np.shape) == 4:
                masks_np = masks_np[:, 0, :, :]
                masks_t = masks_t[:, 0, :, :]

            for i in range(len(scores_np)):
                all_masks.append(masks_np[i].astype(np.uint8))
                all_boxes.append(boxes_np[i].tolist())
                all_scores.append(float(scores_np[i]))
                all_labels_idx.append(label_idx)
                all_masks_tensor.append(masks_t[i])

    result = {
        "masks": all_masks,
        "boxes": all_boxes,
        "scores": all_scores,
        "labels": all_labels_idx,
        "masks_tensor": torch.stack(all_masks_tensor) if all_masks_tensor else torch.empty(0),
    }
    return result, state


def get_image_embedding(state: Any) -> np.ndarray:
    """Extract 256-dim image embedding from SAM3 backbone FPN features.

    Args:
        state: inference_state from set_image().

    Returns:
        1D numpy array of shape (256,).

    """
    backbone_out = state["backbone_out"]
    fpn = backbone_out["backbone_fpn"]
    feat_map = fpn[0]
    if feat_map.dim() == 4:
        feat_map = feat_map[0]
    embedding: np.ndarray = feat_map.mean(dim=(1, 2)).detach().cpu().numpy()
    return embedding


def get_instance_embeddings(state: Any, masks: torch.Tensor) -> np.ndarray:
    """Extract per-instance embeddings via mask-pooling of FPN features.

    Args:
        state: inference_state from set_image().
        masks: Tensor of shape (N, H, W).

    Returns:
        numpy array of shape (N, 256).

    """
    backbone_out = state["backbone_out"]
    feat_map = backbone_out["backbone_fpn"][0]
    if feat_map.dim() == 4:
        feat_map = feat_map[0]
    ch, h_feat, w_feat = feat_map.shape

    if masks.numel() == 0 or masks.shape[0] == 0:
        return np.empty((0, ch))

    masks_float = masks.float().to(feat_map.device)
    if masks_float.dim() == 3:
        masks_float = masks_float.unsqueeze(1)

    masks_resized = f_nn.interpolate(masks_float, size=(h_feat, w_feat), mode="nearest").squeeze(1)

    embeddings = []
    for i in range(masks_resized.shape[0]):
        m = masks_resized[i]
        if m.sum() < 1e-6:
            embeddings.append(torch.zeros(ch, device=feat_map.device))
        else:
            w = m / (m.sum() + 1e-6)
            emb = (feat_map * w).view(ch, -1).sum(dim=1)
            embeddings.append(emb)

    return torch.stack(embeddings).detach().cpu().numpy()


def get_box_embeddings(state: Any, boxes: list[list[float]], image_size: tuple[int, int]) -> np.ndarray:
    """Extract per-instance embeddings via box-crop pooling of FPN features.

    Used for bounding-box modality instead of mask-pooling.

    Args:
        state: inference_state from set_image().
        boxes: List of [x0, y0, x1, y1] in image coordinates.
        image_size: (width, height) of the original image.

    Returns:
        numpy array of shape (N, 256).

    """
    if not boxes:
        backbone_out = state["backbone_out"]
        feat_map = backbone_out["backbone_fpn"][0]
        if feat_map.dim() == 4:
            feat_map = feat_map[0]
        return np.empty((0, feat_map.shape[0]))

    backbone_out = state["backbone_out"]
    feat_map = backbone_out["backbone_fpn"][0]
    if feat_map.dim() == 4:
        feat_map = feat_map[0]
    ch, h_feat, w_feat = feat_map.shape

    img_w, img_h = image_size
    embeddings = []
    for box in boxes:
        x0, y0, x1, y1 = box
        # Scale to feature map
        fx0 = max(0, int(x0 / img_w * w_feat))
        fy0 = max(0, int(y0 / img_h * h_feat))
        fx1 = min(w_feat, int(x1 / img_w * w_feat) + 1)
        fy1 = min(h_feat, int(y1 / img_h * h_feat) + 1)
        if fx1 <= fx0 or fy1 <= fy0:
            embeddings.append(torch.zeros(ch, device=feat_map.device))
        else:
            crop = feat_map[:, fy0:fy1, fx0:fx1]
            embeddings.append(crop.mean(dim=(1, 2)))

    return torch.stack(embeddings).detach().cpu().numpy()


# ── Preview rendering ─────────────────────────────────────────

# Cooler pastel mask fills per class (RGBA, semi-transparent)
_MASK_COLORS = [
    (100, 200, 255, 70),
    (120, 255, 180, 70),
    (200, 160, 255, 70),
    (255, 220, 120, 70),
    (255, 150, 200, 70),
    (150, 240, 230, 70),
    (230, 230, 140, 70),
    (180, 200, 255, 70),
]
# Lighter versions for bounding boxes (same hue, higher brightness)
_BOX_COLORS = [
    (170, 225, 255),
    (180, 255, 215),
    (225, 200, 255),
    (255, 235, 175),
    (255, 195, 225),
    (195, 248, 242),
    (242, 242, 190),
    (215, 225, 255),
]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (100, 200, 255)
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))


def _lighten_rgb(rgb: tuple[int, int, int], factor: float = 0.4) -> tuple[int, int, int]:
    """Lighten an RGB color by blending towards white."""
    return (
        min(255, int(rgb[0] + (255 - rgb[0]) * factor)),
        min(255, int(rgb[1] + (255 - rgb[1]) * factor)),
        min(255, int(rgb[2] + (255 - rgb[2]) * factor)),
    )


def render_preview(
    image: Image.Image,
    predictions: dict[str, Any],
    modality: str = "segmentation",
    label_names: list[str] | None = None,
    label_colors: list[str] | None = None,
) -> str:
    """Render predictions overlaid on image, return base64-encoded PNG.

    Args:
        image: Original PIL image.
        predictions: Output from predict_single_image.
        modality: "segmentation" or "bbox".
        label_names: Class names for legend.
        label_colors: Hex color strings per label (e.g. ["#64c8ff", "#78ffb4"]).

    Returns:
        Base64-encoded PNG string.

    """
    overlay = image.copy().convert("RGBA")
    draw_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)

    masks = predictions["masks"]
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]
    names = predictions.get("label_names", [])

    for i in range(len(scores)):
        cls_idx = labels[i]

        # Use custom colors if provided, otherwise fall back to defaults
        if label_colors and cls_idx < len(label_colors):
            rgb = _hex_to_rgb(label_colors[cls_idx])
            mask_color = (*rgb, 70)
            box_color = _lighten_rgb(rgb)
        else:
            fallback_idx = cls_idx % len(_MASK_COLORS)
            mask_color = _MASK_COLORS[fallback_idx]
            box_color = _BOX_COLORS[fallback_idx]

        name = names[i] if i < len(names) else f"class {labels[i]}"

        if modality == "segmentation" and i < len(masks):
            mask = masks[i]
            mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            colored = Image.new("RGBA", overlay.size, mask_color)
            draw_layer.paste(colored, mask=mask_img)

        # Bounding box in same hue as mask but lighter
        if i < len(boxes):
            x0, y0, x1, y1 = boxes[i]
            draw.rectangle([x0, y0, x1, y1], outline=(*box_color, 200), width=2)
            # Confidence label above the box with black background, white text
            label_text = f"{name} {scores[i]:.2f}"
            text_y = max(0, y0 - 14)
            text_bbox = draw.textbbox((x0 + 2, text_y), label_text)
            draw.rectangle(
                [text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 1],
                fill=(0, 0, 0, 180),
            )
            draw.text((x0 + 2, text_y), label_text, fill=(255, 255, 255, 255))

    result = Image.alpha_composite(overlay, draw_layer)
    result = result.convert("RGB")

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def list_images_in_folder(folder: str, max_count: int = 10000) -> list[str]:
    """List image files in a folder recursively (local, cloud, or aliased).

    Args:
        folder: Path or URL to folder.
        max_count: Maximum number of images to return.

    Returns:
        Sorted list of image paths/URLs.

    """
    from tlc_plugin_sdk.shared.images import list_image_urls

    return list_image_urls(folder, max_count)
