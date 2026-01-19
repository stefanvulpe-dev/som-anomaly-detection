import json
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_pixel_auroc(
    anomaly_maps: np.ndarray,
    ground_truth_masks: np.ndarray,
) -> float:
    """
    Compute pixel-level AUROC.

    Args:
        anomaly_maps: Predicted anomaly score maps (N, H, W) with values in [0, 1].
        ground_truth_masks: Binary ground truth masks (N, H, W) where 1 = anomaly.

    Returns:
        Pixel-level AUROC score.
    """
    # Flatten all pixels
    preds_flat = anomaly_maps.flatten()
    targets_flat = ground_truth_masks.flatten()

    # Handle edge case: all pixels are normal or all are anomalous
    if len(np.unique(targets_flat)) < 2:
        return float("nan")

    return roc_auc_score(targets_flat, preds_flat)


def compute_image_auroc(
    anomaly_maps: np.ndarray,
    ground_truth_masks: np.ndarray,
) -> float:
    """
    Compute image-level AUROC.

    Args:
        anomaly_maps: Predicted anomaly score maps (N, H, W) with values in [0, 1].
        ground_truth_masks: Binary ground truth masks (N, H, W) where 1 = anomaly.

    Returns:
        Image-level AUROC score.
    """
    # Image-level prediction: max anomaly score per image
    image_scores = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(axis=1)

    # Image-level ground truth: 1 if any pixel is anomalous
    image_labels = (
        ground_truth_masks.reshape(ground_truth_masks.shape[0], -1).max(axis=1)
        > 0
    ).astype(int)

    # Handle edge case: all images are normal or all are anomalous
    if len(np.unique(image_labels)) < 2:
        return float("nan")

    return roc_auc_score(image_labels, image_scores)


def compute_iou(
    anomaly_maps: np.ndarray,
    ground_truth_masks: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Compute Intersection over Union (IoU) for anomaly segmentation.

    Args:
        anomaly_maps: Predicted anomaly score maps (N, H, W) with values in [0, 1].
        ground_truth_masks: Binary ground truth masks (N, H, W) where 1 = anomaly.
        threshold: Threshold to binarize predictions.

    Returns:
        IoU score.
    """
    # Binarize predictions
    preds_binary = (anomaly_maps >= threshold).astype(int)
    targets_binary = (ground_truth_masks > 0).astype(int)

    # Compute intersection and union
    intersection = np.logical_and(preds_binary, targets_binary).sum()
    union = np.logical_or(preds_binary, targets_binary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def compute_best_iou(
    anomaly_maps: np.ndarray,
    ground_truth_masks: np.ndarray,
    thresholds: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Find the best IoU across different thresholds.

    Args:
        anomaly_maps: Predicted anomaly score maps (N, H, W) with values in [0, 1].
        ground_truth_masks: Binary ground truth masks (N, H, W) where 1 = anomaly.
        thresholds: Array of thresholds to try. If None, uses linspace(0.1, 0.9, 17).

    Returns:
        Tuple of (best_iou, best_threshold).
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    best_iou = 0.0
    best_threshold = 0.5

    for thresh in thresholds:
        iou = compute_iou(anomaly_maps, ground_truth_masks, threshold=thresh)
        if iou > best_iou:
            best_iou = iou
            best_threshold = thresh

    return best_iou, best_threshold


def compute_all_metrics(
    anomaly_maps: np.ndarray,
    ground_truth_masks: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all anomaly detection metrics.

    Args:
        anomaly_maps: Predicted anomaly score maps (N, H, W) with values in [0, 1].
        ground_truth_masks: Binary ground truth masks (N, H, W) where 1 = anomaly.
        iou_threshold: Threshold for IoU computation.

    Returns:
        Dictionary with all computed metrics.
    """
    pixel_auroc = compute_pixel_auroc(anomaly_maps, ground_truth_masks)
    image_auroc = compute_image_auroc(anomaly_maps, ground_truth_masks)
    iou = compute_iou(anomaly_maps, ground_truth_masks, threshold=iou_threshold)
    best_iou, best_thresh = compute_best_iou(anomaly_maps, ground_truth_masks)

    return {
        "pixel_auroc": pixel_auroc,
        "image_auroc": image_auroc,
        "iou": iou,
        "best_iou": best_iou,
        "best_iou_threshold": best_thresh,
    }


def print_metrics(metrics: Dict[str, float], category: str = None) -> None:
    """
    Pretty print the computed metrics.

    Args:
        metrics: Dictionary of metric names to values.
        category: Optional category name for the header.
    """
    header = f"Metrics for '{category}'" if category else "Metrics"
    print(f"\n{'=' * 50}")
    print(f"📊 {header}")
    print("=" * 50)
    print(f"  Pixel AUROC:      {metrics['pixel_auroc']:.4f}")
    print(f"  Image AUROC:      {metrics['image_auroc']:.4f}")
    print(f"  IoU (@0.5):       {metrics['iou']:.4f}")
    print(
        f"  Best IoU:         {metrics['best_iou']:.4f} "
        f"(threshold={metrics['best_iou_threshold']:.2f})"
    )
    print("=" * 50)


def save_metrics_to_json(
    metrics: Dict[str, float],
    output_dir: str,
    category: str = None,
    filename: str = None,
) -> str:
    """
    Save metrics to a JSON file.

    Args:
        metrics: Dictionary of metric names to values.
        output_dir: Directory to save the JSON file.
        category: Optional category name to include in the filename.
        filename: Optional custom filename. If None, generates one with timestamp.

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if category:
            filename = f"metrics_{category}_{timestamp}.json"
        else:
            filename = f"metrics_{timestamp}.json"

    filepath = os.path.join(output_dir, filename)

    # Prepare data with metadata
    data = {
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "metrics": metrics,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def save_all_metrics_to_json(
    all_metrics: Dict[str, Dict[str, float]],
    output_dir: str,
    filename: str = "metrics_all_categories.json",
) -> str:
    """
    Save metrics for all categories to a single JSON file.

    Args:
        all_metrics: Dictionary mapping category names to their metrics.
        output_dir: Directory to save the JSON file.
        filename: Filename for the JSON file.

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Compute average metrics across categories
    avg_metrics = {}
    metric_keys = ["pixel_auroc", "image_auroc", "iou", "best_iou"]

    for key in metric_keys:
        values = [
            m[key]
            for m in all_metrics.values()
            if not np.isnan(m.get(key, float("nan")))
        ]
        avg_metrics[key] = float(np.mean(values)) if values else float("nan")

    # Prepare data with metadata
    data = {
        "timestamp": datetime.now().isoformat(),
        "num_categories": len(all_metrics),
        "average_metrics": avg_metrics,
        "per_category_metrics": all_metrics,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n💾 Metrics saved to: {filepath}")
    return filepath
