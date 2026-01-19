"""
Self-Organizing Map (SOM) Visualization Module.

This module provides publication-quality visualizations for Self-Organizing Maps,
using hexagonal cell heatmaps as is standard in the SOM literature.
"""

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import fcluster, linkage


def compute_u_matrix(som_weights: np.ndarray) -> np.ndarray:
    """
    Compute the Unified Distance Matrix (U-Matrix) for a SOM.

    The U-Matrix shows the distances between neighboring neurons, revealing
    cluster boundaries in the SOM.

    Args:
        som_weights: SOM weight vectors of shape (X, Y, D).

    Returns:
        U-Matrix of shape (X, Y) with average distances to neighbors.
    """
    x, y, _ = som_weights.shape
    u_matrix = np.zeros((x, y))

    for i in range(x):
        for j in range(y):
            neighbors = []
            # Check all 8 neighbors (including diagonals)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < x and 0 <= nj < y:
                        dist = np.linalg.norm(
                            som_weights[i, j] - som_weights[ni, nj]
                        )
                        neighbors.append(dist)
            u_matrix[i, j] = np.mean(neighbors) if neighbors else 0

    return u_matrix


def compute_neuron_activations(
    som_weights: np.ndarray,
    data: np.ndarray,
) -> np.ndarray:
    """
    Compute how many data points are assigned to each neuron (hit map).

    Args:
        som_weights: SOM weight vectors of shape (X, Y, D).
        data: Training data of shape (N, D).

    Returns:
        Hit map of shape (X, Y) showing activation counts per neuron.
    """
    x, y, d = som_weights.shape
    weights_flat = som_weights.reshape(-1, d)

    # Find BMU (Best Matching Unit) for each data point
    hit_map = np.zeros((x, y))
    for point in data:
        distances = np.linalg.norm(weights_flat - point, axis=1)
        bmu_idx = np.argmin(distances)
        bmu_x, bmu_y = bmu_idx // y, bmu_idx % y
        hit_map[bmu_x, bmu_y] += 1

    return hit_map


def _draw_hexagonal_grid(
    ax: plt.Axes,
    values: np.ndarray,
    cmap: str = "viridis",
    colorbar_label: str = "Value",
    show_colorbar: bool = True,
) -> None:
    """
    Draw a hexagonal grid heatmap on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        values: 2D array of values to visualize (X, Y).
        cmap: Colormap name.
        colorbar_label: Label for the colorbar.
        show_colorbar: Whether to show the colorbar.
    """
    x, y = values.shape

    # Normalize values for coloring
    norm = Normalize(vmin=values.min(), vmax=values.max())
    colormap = cm.get_cmap(cmap)

    # Hexagon parameters - adjust size based on grid dimensions
    hex_size = 0.58

    for i in range(x):
        for j in range(y):
            # Offset for hexagonal layout (odd rows shifted)
            offset_x = 0.5 * hex_size * 1.732 if j % 2 == 1 else 0
            center_x = i * hex_size * 1.732 + offset_x
            center_y = j * hex_size * 1.5

            color = colormap(norm(values[i, j]))
            hexagon = RegularPolygon(
                (center_x, center_y),
                numVertices=6,
                radius=hex_size,
                orientation=0,  # Flat-top hexagons
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.add_patch(hexagon)

    # Set axis limits with padding
    ax.set_xlim(-hex_size, x * hex_size * 1.732 + hex_size)
    ax.set_ylim(-hex_size, y * hex_size * 1.5 + hex_size)
    ax.set_aspect("equal")
    ax.axis("off")

    if show_colorbar:
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label(colorbar_label, fontsize=10)


def plot_hexagonal_u_matrix(
    som_weights: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "U-Matrix",
    cmap: str = "viridis_r",
    show_colorbar: bool = True,
) -> plt.Axes:
    """
    Plot the U-Matrix as a hexagonal grid showing cluster boundaries.

    Args:
        som_weights: SOM weight vectors of shape (X, Y, D).
        ax: Matplotlib axes to plot on. If None, creates new figure.
        title: Plot title.
        cmap: Colormap name.
        show_colorbar: Whether to show colorbar.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    u_matrix = compute_u_matrix(som_weights)
    _draw_hexagonal_grid(
        ax,
        u_matrix,
        cmap=cmap,
        colorbar_label="Distance",
        show_colorbar=show_colorbar,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")

    return ax


def plot_hexagonal_weight_variance(
    som_weights: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Weight Variance",
    cmap: str = "plasma",
    show_colorbar: bool = True,
) -> plt.Axes:
    """
    Plot variance of weights for each neuron as hexagonal grid.

    Args:
        som_weights: SOM weight vectors of shape (X, Y, D).
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
        cmap: Colormap name.
        show_colorbar: Whether to show colorbar.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    variance_map = np.var(som_weights, axis=2)
    _draw_hexagonal_grid(
        ax,
        variance_map,
        cmap=cmap,
        colorbar_label="Variance",
        show_colorbar=show_colorbar,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")

    return ax


def plot_hexagonal_hit_map(
    som_weights: np.ndarray,
    train_data: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Neuron Activation Map",
    cmap: str = "plasma",
    show_colorbar: bool = True,
) -> plt.Axes:
    """
    Plot hit map (neuron activations) as hexagonal grid.

    Args:
        som_weights: SOM weight vectors of shape (X, Y, D).
        train_data: Training data of shape (N, D).
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
        cmap: Colormap name.
        show_colorbar: Whether to show colorbar.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    hit_map = compute_neuron_activations(som_weights, train_data)
    _draw_hexagonal_grid(
        ax,
        hit_map,
        cmap=cmap,
        colorbar_label="Hit Count",
        show_colorbar=show_colorbar,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")

    return ax


def plot_hexagonal_cluster_assignments(
    som_weights: np.ndarray,
    n_clusters: int = 5,
    ax: Optional[plt.Axes] = None,
    method: str = "ward",
    title: str = "Cluster Assignments",
    cmap: str = "tab10",
    show_colorbar: bool = True,
) -> Tuple[plt.Axes, np.ndarray]:
    """
    Plot cluster assignments for SOM neurons as hexagonal grid.

    Args:
        som_weights: SOM weight vectors of shape (X, Y, D).
        n_clusters: Number of clusters.
        ax: Matplotlib axes. If None, creates new figure.
        method: Linkage method for hierarchical clustering.
        title: Plot title.
        cmap: Colormap for clusters.
        show_colorbar: Whether to show colorbar.

    Returns:
        Tuple of (matplotlib axes, cluster labels array).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    x, y, d = som_weights.shape
    weights_flat = som_weights.reshape(-1, d)

    # Hierarchical clustering
    linkage_matrix = linkage(weights_flat, method=method)
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    cluster_map = cluster_labels.reshape(x, y)

    _draw_hexagonal_grid(
        ax,
        cluster_map.astype(float),
        cmap=cmap,
        colorbar_label="Cluster ID",
        show_colorbar=show_colorbar,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")

    return ax, cluster_map


def plot_distance_distribution(
    som_weights: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "U-Matrix Distance Distribution",
) -> plt.Axes:
    """
    Plot histogram of U-Matrix distances.

    Args:
        som_weights: SOM weight vectors of shape (X, Y, D).
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    u_matrix = compute_u_matrix(som_weights)
    ax.hist(
        u_matrix.flatten(),
        bins=30,
        color="steelblue",
        edgecolor="white",
        alpha=0.8,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Distance", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def _setup_publication_style() -> None:
    """Set publication-quality matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _save_figure(
    fig: plt.Figure,
    out_dir: str,
    filename: str,
    dpi: int = 300,
    save_pdf: bool = True,
) -> str:
    """Save figure as PNG and optionally PDF."""
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"{filename}.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    if save_pdf:
        pdf_path = os.path.join(out_dir, f"{filename}.pdf")
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    plt.close(fig)
    return png_path


def export_som_plots(
    som_weights: np.ndarray,
    category: str,
    out_dir: str,
    train_data: Optional[np.ndarray] = None,
    n_clusters: int = 5,
    dpi: int = 300,
    figsize: Tuple[int, int] = (8, 6),
) -> dict:
    """
    Export individual SOM visualization plots for a single category.

    This function generates separate publication-quality figures:
    - U-Matrix showing cluster boundaries
    - Weight Variance map
    - Cluster Assignments
    - 2D PCA projection of neurons
    - Neuron Activation Map (hit map)
    - Distance Distribution histogram

    Args:
        som_weights: SOM weight vectors of shape (X, Y, D).
        category: Category name (e.g., 'toothbrush').
        out_dir: Output directory for saving the plots.
        train_data: Optional training data for hit map visualization.
        n_clusters: Number of clusters for clustering visualization.
        dpi: DPI for saved figures.
        figsize: Figure size for each plot.

    Returns:
        Dictionary mapping plot names to their file paths.
    """
    _setup_publication_style()

    # Create category-specific subdirectory
    category_dir = os.path.join(out_dir, f"som_plots_{category}")
    os.makedirs(category_dir, exist_ok=True)

    saved_paths = {}

    # 1. U-Matrix
    fig, ax = plt.subplots(figsize=figsize)
    plot_hexagonal_u_matrix(
        som_weights, ax=ax, title="U-Matrix", cmap="viridis_r"
    )
    saved_paths["u_matrix"] = _save_figure(
        fig, category_dir, f"{category}_u_matrix", dpi=dpi
    )

    # 2. Weight Variance
    fig, ax = plt.subplots(figsize=figsize)
    plot_hexagonal_weight_variance(
        som_weights, ax=ax, title="Weight Variance", cmap="plasma"
    )
    saved_paths["weight_variance"] = _save_figure(
        fig, category_dir, f"{category}_weight_variance", dpi=dpi
    )

    # 3. Cluster Assignments
    fig, ax = plt.subplots(figsize=figsize)
    plot_hexagonal_cluster_assignments(
        som_weights,
        n_clusters=n_clusters,
        ax=ax,
        title=f"Cluster Assignments (k={n_clusters})",
        cmap="tab10",
    )
    saved_paths["cluster_assignments"] = _save_figure(
        fig, category_dir, f"{category}_cluster_assignments", dpi=dpi
    )

    # 4. Neuron Activation Map (if train_data provided)
    if train_data is not None:
        fig, ax = plt.subplots(figsize=figsize)
        plot_hexagonal_hit_map(
            som_weights,
            train_data,
            ax=ax,
            title="Neuron Activation Map",
            cmap="jet",
        )
        saved_paths["activation_map"] = _save_figure(
            fig, category_dir, f"{category}_activation_map", dpi=dpi
        )

    # 5. Distance Distribution
    fig, ax = plt.subplots(figsize=figsize)
    plot_distance_distribution(
        som_weights, ax=ax, title="U-Matrix Distance Distribution"
    )
    saved_paths["distance_distribution"] = _save_figure(
        fig, category_dir, f"{category}_distance_distribution", dpi=dpi
    )

    return saved_paths


def export_som_plot(
    som_weights: np.ndarray,
    category: str,
    out_dir: str,
    train_data: Optional[np.ndarray] = None,
    n_clusters: int = 5,
    dpi: int = 300,
) -> str:
    """
    Export SOM visualizations for a single category as separate plots.

    Args:
        som_weights: SOM weight vectors of shape (X, Y, D).
        category: Category name (e.g., 'toothbrush').
        out_dir: Output directory for saving the plots.
        train_data: Optional training data for hit map visualization.
        n_clusters: Number of clusters for clustering visualization.
        dpi: DPI for saved figures.

    Returns:
        Path to the output directory containing all plots.
    """
    export_som_plots(
        som_weights=som_weights,
        category=category,
        out_dir=out_dir,
        train_data=train_data,
        n_clusters=n_clusters,
        dpi=dpi,
    )

    # Return the directory containing all plots
    return os.path.join(out_dir, f"som_plots_{category}")
