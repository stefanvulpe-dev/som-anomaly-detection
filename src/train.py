import os
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.backbone import get_backbone, resolve_selected_indexes
from src.config import Config
from src.data import make_dataloaders
from src.img_helpers import (
    apply_colormap_on_image,
    apply_contours_on_image,
    create_comparison_visualization,
    normalize_heatmap,
    upsample_and_smooth,
)
from src.metrics import (
    compute_all_metrics,
    print_metrics,
    save_all_metrics_to_json,
    save_metrics_to_json,
)
from src.som_visualization import export_som_plot
from src.somad import (
    cluster_embeddings,
    compute_anomaly_scores,
    extract_embeddings,
    extract_features,
)


def train(cfg: Config, category: str):
    model, image_transform = get_backbone(cfg)

    train_loader, _ = make_dataloaders(
        cfg=cfg,
        image_transform=image_transform,
        category=category,
        split="train",
    )

    train_embeddings = extract_embeddings(
        model=model,
        dataloader=train_loader,
        selected_indexes=resolve_selected_indexes(cfg),
        max_images=cfg.max_train_images,
        device=cfg.device,
    )

    _, _, H, W = train_embeddings.shape
    init_weights = train_embeddings.mean(dim=0).permute(1, 2, 0).cpu().numpy()

    train_embeddings = train_embeddings.permute(0, 2, 3, 1).reshape(
        -1, train_embeddings.size(1)
    )
    train_embeddings = train_embeddings.cpu().numpy()

    means, cov, som_weights = cluster_embeddings(
        feature_embeddings=train_embeddings,
        cfg=cfg,
        som_x=W,
        som_y=H,
        init_weights=init_weights,
    )

    return means, cov, som_weights, train_embeddings


def infer(
    category: str,
    means: np.ndarray,
    cov: np.ndarray,
    som_weights: np.ndarray,
    cfg: Config,
) -> dict:
    model, image_transform = get_backbone(cfg)

    test_loader, _ = make_dataloaders(
        cfg=cfg,
        image_transform=image_transform,
        category=category,
        split="test",
    )

    # Setup output directories once
    heatmap_dir, contour_dir, comparison_dir = setup_output_dirs(cfg)

    # Collect all predictions and ground truths for metrics
    all_anomaly_maps = []
    all_ground_truths = []

    for idx, batch in enumerate(
        tqdm(
            test_loader,
            desc=f"Inference on test set for category '{category}'",
            leave=False,
        )
    ):
        images = batch[0]
        mask_tensor = batch[1]
        img_paths = batch[4]

        test_embeddings = extract_features(model, images, device=cfg.device)
        test_embeddings = test_embeddings.index_select(
            dim=1,
            index=torch.tensor(
                resolve_selected_indexes(cfg), device=test_embeddings.device
            ),
        )
        B, _, H, W = test_embeddings.shape
        test_embeddings = (
            test_embeddings.permute(0, 2, 3, 1)
            .reshape(-1, test_embeddings.size(1))
            .cpu()
            .numpy()
        )

        scores = compute_anomaly_scores(
            embeddings=test_embeddings,
            means=means,
            covs_inv=cov,
            som_weights=som_weights,
            k=cfg.k,
        )
        scores = scores.reshape(B, H, W)

        # Upsample scores and masks to original image size for metrics
        anomaly_maps, gt_masks = prepare_for_metrics(
            scores=scores,
            mask_tensor=mask_tensor,
            target_size=(cfg.img_size, cfg.img_size),
        )
        all_anomaly_maps.append(anomaly_maps)
        all_ground_truths.append(gt_masks)

        # Visualize on the fly for this batch (limited number)
        if idx * test_loader.batch_size < cfg.max_visualize_count:
            visualize_results(
                cfg=cfg,
                scores=scores,
                mask_tensor=mask_tensor,
                img_paths=img_paths,
                heatmap_dir=heatmap_dir,
                contour_dir=contour_dir,
                comparison_dir=comparison_dir,
            )

    # Compute metrics on all test images
    all_anomaly_maps = np.concatenate(all_anomaly_maps, axis=0)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)

    metrics = compute_all_metrics(all_anomaly_maps, all_ground_truths)
    print_metrics(metrics, category=category)
    print(f"\n✅ Results saved to '{cfg.out_dir}':")

    return metrics


def prepare_for_metrics(
    scores: np.ndarray,
    mask_tensor: torch.Tensor,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare anomaly scores and ground truth masks for metric computation.

    Args:
        scores: Anomaly score maps of shape (B, H, W).
        mask_tensor: Ground truth masks as a tensor of shape (B, 1, H, W).
        target_size: Target size to upsample scores to match masks.

    Returns:
        Tuple of (normalized_anomaly_maps, ground_truth_masks) both of shape (B, H, W).
    """
    B = scores.shape[0]
    anomaly_maps = []

    for i in range(B):
        # Upsample and smooth the score map
        upsampled = upsample_and_smooth(
            scores[i], out_size=target_size, sigma=4
        )
        # Normalize to [0, 1]
        normalized = normalize_heatmap(upsampled)
        anomaly_maps.append(normalized)

    anomaly_maps = np.stack(anomaly_maps, axis=0)

    # Process ground truth masks
    gt_masks = mask_tensor.squeeze(1).numpy()  # (B, H, W)
    # Binarize: ensure values are 0 or 1
    gt_masks = (gt_masks > 0).astype(np.float32)

    return anomaly_maps, gt_masks


def setup_output_dirs(cfg: Config) -> tuple[str, str, str]:
    """Create output subdirectories and return their paths."""
    comparison_dir = os.path.join(cfg.out_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    return None, None, comparison_dir


def visualize_results(
    scores: np.ndarray,
    img_paths: list[str],
    mask_tensor: torch.Tensor,
    heatmap_dir: str,
    contour_dir: str,
    comparison_dir: str,
    cfg: Config,
) -> None:
    """
    Generate visualizations for a batch of images on the fly.

    Args:
        scores: Anomaly score maps of shape (B, H, W).
        img_paths: List of image paths corresponding to scores.
        mask_tensor: Ground truth masks as a tensor of shape (B, 1, H, W).
        heatmap_dir: Directory to save heatmap overlays.
        contour_dir: Directory to save contour visualizations.
        comparison_dir: Directory to save comparison panels.
        cfg: Configuration object.

    Returns:
        None
    """

    for idx, img_path in enumerate(img_paths):
        anomaly_map = scores[idx]
        norm = normalize_heatmap(
            anomaly_map, vmin=anomaly_map.min(), vmax=anomaly_map.max()
        )
        basename = os.path.splitext(os.path.basename(img_path))[0]
        dir_name = os.path.dirname(img_path).split(os.sep)[-3]
        basename = f"{dir_name}_{basename}"

        # 1. Heatmap overlay
        if heatmap_dir is not None:
            heatmap_path = os.path.join(heatmap_dir, f"{basename}_heatmap.png")
            apply_colormap_on_image(
                img_path,
                norm,
                out_path=heatmap_path,
                alpha=0.5,
                threshold=cfg.contour_threshold,
            )

        # 2. Contour visualization
        if contour_dir is not None:
            contour_path = os.path.join(contour_dir, f"{basename}_contours.png")
            apply_contours_on_image(
                img_path,
                norm,
                out_path=contour_path,
                threshold=cfg.contour_threshold,
                contour_color=(255, 0, 0),
                fill_alpha=0.3,
                contour_thickness=3,
                min_area=100,
            )

        # 3. Comparison panel
        if comparison_dir is not None:
            comparison_path = os.path.join(
                comparison_dir, f"{basename}_comparison.png"
            )
            create_comparison_visualization(
                img_path,
                norm,
                out_path=comparison_path,
                threshold=cfg.contour_threshold,
                show_heatmap=True,
                show_contours=True,
                ground_truth_mask=mask_tensor[idx].squeeze(0).numpy(),
            )


def run_for_one_category(cfg: Config):
    means, cov, som_weights, train_data = train(cfg, category="toothbrush")

    # Export SOM visualization
    som_plot_path = export_som_plot(
        som_weights=som_weights,
        category="toothbrush",
        out_dir=cfg.out_dir,
        train_data=train_data,
        n_clusters=5,
        dpi=300,
    )
    print(f"📊 SOM visualization saved to: {som_plot_path}")

    metrics = infer(
        category="toothbrush",
        means=means,
        cov=cov,
        som_weights=som_weights,
        cfg=cfg,
    )
    save_metrics_to_json(metrics, cfg.out_dir, category="toothbrush")


def run_for_all_categories(cfg: Config):
    categories = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]
    all_metrics = {}
    for category in tqdm(categories, desc="Processing categories", leave=False):
        means, cov, som_weights, train_data = train(cfg, category=category)

        # Export SOM visualization for each category
        som_plot_path = export_som_plot(
            som_weights=som_weights,
            category=category,
            out_dir=cfg.out_dir,
            train_data=train_data,
            n_clusters=5,
            dpi=300,
        )
        print(
            f"📊 SOM visualization for '{category}' saved to: {som_plot_path}"
        )

        metrics = infer(
            category=category,
            means=means,
            cov=cov,
            som_weights=som_weights,
            cfg=cfg,
        )
        all_metrics[category] = metrics

    save_all_metrics_to_json(all_metrics, cfg.out_dir)
