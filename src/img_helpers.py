from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter


def upsample_and_smooth(
    score_map: np.ndarray,
    out_size: Tuple[int, int] = (224, 224),
    sigma: float = 4,
) -> np.ndarray:
    """
    Upsample the score map to the desired output size and apply Gaussian smoothing.

    Args:
        score_map (np.ndarray): The input score map to be upsampled and smoothed.
        out_size (Tuple[int, int]): The desired output size (height, width).
        sigma (float): The standard deviation for Gaussian kernel.

    Returns:
        np.ndarray: The upsampled and smoothed score map.
    """
    score_map_torch = (
        torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0).float()
    )
    up = F.interpolate(
        score_map_torch, size=out_size, mode="bilinear", align_corners=False
    )
    up = up.squeeze().numpy()
    sm = gaussian_filter(up, sigma=sigma)
    return sm


def normalize_heatmap(
    heatmap: np.ndarray, vmin: float = None, vmax: float = None
) -> np.ndarray:
    """
    Normalize the heatmap to the range [0, 1].

    Args:
        heatmap (np.ndarray): The input heatmap to be normalized.
        vmin (float, optional): Minimum value for normalization. If None, uses the min of the heatmap.
        vmax (float, optional): Maximum value for normalization. If None, uses the max of the heatmap.

    Returns:
        np.ndarray: The normalized heatmap.
    """
    if vmin is None:
        vmin = float(np.min(heatmap))
    if vmax is None:
        vmax = float(np.max(heatmap))
    if vmax - vmin < 1e-8:
        return np.zeros_like(heatmap)
    return (heatmap - vmin) / (vmax - vmin)


def apply_colormap_on_image(
    img_path: str,
    heatmap_norm: np.ndarray,
    out_path: str = None,
    alpha: float = 0.5,
    cmap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Apply a colormap on the normalized heatmap and overlay it on the original image.

    Args:
        img_path (str): Path to the original image.
        heatmap_norm (np.ndarray): Normalized heatmap (values in [0, 1]).
        out_path (str, optional): Path to save the overlay image. If None, the image is not saved.
        alpha (float): Transparency factor for the overlay.
        cmap (int): OpenCV colormap to apply.

    Returns:
        np.ndarray: The overlay image as a numpy array.
    """
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap_uint8 = (heatmap_norm * 255).astype("uint8")
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cmap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = cv2.resize(
        heatmap_color, (img_rgb.shape[1], img_rgb.shape[0])
    )
    overlay = (img_rgb * (1.0 - alpha) + heatmap_color * alpha).astype("uint8")
    if out_path is not None:
        plt.imsave(out_path, overlay)
    return overlay


def apply_contours_on_image(
    img_path: str,
    heatmap_norm: np.ndarray,
    out_path: str = None,
    threshold: float = 0.5,
    contour_color: tuple = (255, 0, 0),
    fill_alpha: float = 0.3,
    contour_thickness: int = 3,
    min_area: int = 100,
) -> Tuple[np.ndarray, int]:
    """
    Draw contours around anomalous regions on the original image.

    Args:
        img_path: Path to original image
        heatmap_norm: Normalized heatmap (0-1 range)
        out_path: Where to save result (optional)
        threshold: Anomaly threshold (0-1), higher = only strong anomalies
        contour_color: RGB color for contour lines (default: red)
        fill_alpha: Transparency for filled regions (0=transparent, 1=opaque)
        contour_thickness: Line thickness in pixels
        min_area: Minimum contour area to draw (filters noise)

    Returns:
        result_img: Image with contours drawn
    """
    # Load original image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(
        heatmap_norm,
        (img_rgb.shape[1], img_rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # Create binary mask from threshold
    mask = (heatmap_resized > threshold).astype(np.uint8) * 255

    # Optional: morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Create result image
    result = img_rgb.copy()

    # Filter contours by area and draw
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if len(valid_contours) > 0:
        # Optional: fill contours with semi-transparent color
        if fill_alpha > 0:
            overlay = img_rgb.copy()
            cv2.drawContours(overlay, valid_contours, -1, contour_color, -1)
            result = cv2.addWeighted(
                overlay, fill_alpha, result, 1 - fill_alpha, 0
            )

        # Draw contour boundaries
        cv2.drawContours(
            result, valid_contours, -1, contour_color, contour_thickness
        )

    # Save if path provided
    if out_path is not None:
        plt.imsave(out_path, result)

    return result, len(valid_contours)


def create_comparison_visualization(
    img_path: str,
    heatmap_norm: np.ndarray,
    out_path: str = None,
    threshold: float = 0.5,
    show_heatmap: bool = True,
    show_contours: bool = True,
    ground_truth_mask: np.ndarray = None,
) -> plt.Figure:
    """
    Create side-by-side visualization with original, heatmap, contours, and ground truth.

    Args:
        img_path: Path to original image
        heatmap_norm: Normalized heatmap (0-1)
        out_path: Where to save (optional)
        threshold: Threshold for contours
        show_heatmap: Include heatmap overlay panel
        show_contours: Include contour panel
        ground_truth_mask: Ground truth mask array (optional). If provided,
            will be displayed as an additional panel.

    Returns:
        combined_img: Multi-panel visualization
    """
    # Load original
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    panels = [img_rgb]
    titles = ["Original"]

    # Add ground truth mask if provided
    if ground_truth_mask is not None:
        # Resize mask to match image dimensions if needed
        if ground_truth_mask.shape[:2] != img_rgb.shape[:2]:
            gt_resized = cv2.resize(
                ground_truth_mask.astype(np.float32),
                (img_rgb.shape[1], img_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            gt_resized = ground_truth_mask.astype(np.float32)

        # Normalize mask to 0-1 range if needed
        if gt_resized.max() > 1:
            gt_resized = gt_resized / 255.0

        # Create RGB visualization of the mask
        gt_display = np.zeros_like(img_rgb)
        gt_display[:, :, 0] = (gt_resized * 255).astype(np.uint8)  # Red channel
        panels.append(gt_display)
        titles.append("Ground Truth")

    # Add heatmap overlay
    if show_heatmap:
        heatmap_overlay = apply_colormap_on_image(
            img_path, heatmap_norm, out_path=None, alpha=0.5
        )
        panels.append(heatmap_overlay)
        titles.append("Heatmap")

    # Add contours
    if show_contours:
        contour_img, num_regions = apply_contours_on_image(
            img_path, heatmap_norm, out_path=None, threshold=threshold
        )
        panels.append(contour_img)
        titles.append(f"Contours ({num_regions} regions)")

    # Create figure
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))

    if len(panels) == 1:
        axes = [axes]

    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig
