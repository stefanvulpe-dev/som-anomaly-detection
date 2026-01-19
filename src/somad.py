from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from minisom import MiniSom
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config


def extract_features(
    model: torch.nn.Module, images: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """
    Extract features from images using the given model.

    Args:
        model (torch.nn.Module): The neural network model to extract features.
        images (torch.Tensor): A batch of images (B x C x H x W).
        device (str): The device to run the model on ("cpu" or "cuda").

    Returns:
        torch.Tensor: A tensor containing the extracted features (B, C, H, W).
    """
    with torch.no_grad():
        images = images.to(device)
        f1, f2, f3 = model(images)
        _, _, H, W = f1.shape
        f2_u = F.interpolate(
            f2, size=(H, W), mode="bilinear", align_corners=False
        )
        f3_u = F.interpolate(
            f3, size=(H, W), mode="bilinear", align_corners=False
        )
        concat = torch.cat([f1, f2_u, f3_u], dim=1)
        return concat


def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    selected_indexes: list[int],
    max_images: int = -1,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Extract embeddings from a dataset using the given model.

    Args:
        model (torch.nn.Module): The neural network model to extract features.
        dataloader (DataLoader): DataLoader for the dataset.
        selected_indexes (list[int]): List of feature dimensions to select.
        max_images (int): Maximum number of images to process (-1 for all).
        device (str): The device to run the model on ("cpu" or "cuda").

    Returns:
        torch.Tensor: A tensor containing feature embeddings for the entire dataset.
    """
    all_embeddings = []

    for idx, batch in enumerate(
        tqdm(
            dataloader,
            desc=f"Extracting features for {dataloader.dataset.split} set",
            leave=False,
        )
    ):
        if max_images != -1 and idx * dataloader.batch_size >= max_images:
            break
        images = batch[0]
        features = extract_features(model, images, device=device)

        features = features.index_select(
            dim=1,
            index=torch.tensor(selected_indexes, device=features.device),
        )
        all_embeddings.append(features)

    return torch.cat(all_embeddings, dim=0).to(device)


def cluster_embeddings(
    feature_embeddings: np.ndarray,
    som_x: int,
    som_y: int,
    init_weights: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster feature embeddings using a Self-Organizing Map (SOM) and compute
    covariance matrices for each cluster.

    Args:
        feature_embeddings (np.ndarray): The feature embeddings to cluster.
        som_x (int): The width of the SOM grid.
        som_y (int): The height of the SOM grid.
        init_weights (np.ndarray): Initial weights for the SOM.
        cfg (Config): Configuration parameters for the SOM.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of means, covariance matrices,
        and SOM weights for each SOM node.
    """
    som = MiniSom(
        x=som_x,
        y=som_y,
        input_len=feature_embeddings.shape[1],
        sigma=cfg.som_sigma,
        learning_rate=cfg.learning_rate,
        random_seed=cfg.seed,
    )
    som._weights = init_weights
    som.train(
        data=feature_embeddings,
        num_iteration=cfg.som_iterations,
        use_epochs=False,
        verbose=False,
    )
    weights = som.get_weights()
    win_map = som.win_map(feature_embeddings)
    dim = weights.shape[2]
    reg = 0.01 * np.eye(dim)

    covs, means = [], []
    with tqdm(
        total=som_x * som_y, desc="Computing covariances", leave=False
    ) as pbar:
        for i in range(som_x):
            for j in range(som_y):
                samples = np.array(win_map[(i, j)])
                if len(samples) > 1:
                    diffs = samples - weights[i, j]
                    cov = (diffs.T @ diffs) / (len(samples) - 1)
                    mean = np.mean(samples, axis=0)
                else:
                    cov = np.zeros((dim, dim))
                    mean = np.zeros((dim,))

                means.append(mean)
                covs.append(np.linalg.inv(cov + reg))
                pbar.update(1)

    return np.array(means), np.array(covs), som.get_weights()


def _mahalanobis_distance(
    x: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray
) -> float:
    """
    Compute the Mahalanobis distance between a point and a distribution.

    Args:
        x (np.ndarray): The data point.
        mean (np.ndarray): The mean of the distribution.
        cov_inv (np.ndarray): The inverse covariance matrix of the distribution.

    Returns:
        float: The Mahalanobis distance.
    """
    diff = x - mean
    dist = np.sqrt(diff.T @ cov_inv @ diff)
    return dist


def compute_anomaly_scores(
    embeddings: np.ndarray,
    means: np.ndarray,
    covs_inv: np.ndarray,
    som_weights: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Compute anomaly scores for feature vectors based on Mahalanobis distance
    to the k nearest SOM nodes.

    Args:
        embeddings (np.ndarray): The feature vectors (N x D).
        means (np.ndarray): The means of the SOM nodes (M x D).
        covs_inv (np.ndarray): The inverse covariance matrices of the SOM nodes (M x D x D).
        som_weights (np.ndarray): The weights of the SOM nodes (X x Y x D).
        k (int): The number of nearest neighbors to consider.

    Returns:
        np.ndarray: An array of anomaly scores for each feature vector.
    """
    # Flatten SOM weights from (X, Y, D) to (X*Y, D) to match means/covs_inv indexing
    som_x, som_y, dim = som_weights.shape
    weights_flat = som_weights.reshape(som_x * som_y, dim)

    # Get k nearest neurons for each feature vector
    # embeddings: (N, D), weights_flat: (M, D) where M = X*Y
    distances = np.linalg.norm(
        embeddings[:, np.newaxis, :] - weights_flat[np.newaxis, :, :],
        axis=2,
    )
    knn_indices = np.argsort(distances, axis=1)[:, :k]

    # Compute anomaly scores based on Mahalanobis distance
    scores = np.zeros(embeddings.shape[0])
    for i in range(embeddings.shape[0]):
        point = embeddings[i]
        knn_means = means[knn_indices[i]]
        knn_cov_invs = covs_inv[knn_indices[i]]
        dists = [
            _mahalanobis_distance(point, mean, cov_inv)
            for mean, cov_inv in zip(knn_means, knn_cov_invs)
        ]
        scores[i] = np.min(dists)

    return scores
