import argparse

import numpy as np
import torch

from src.config import Config
from src.train import run_for_all_categories


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train and infer SOM model")
    parser.add_argument(
        "--backbone",
        type=str,
        default="convnextv2_nano.fcmae_ft_in22k_in1k",
        help="Backbone model name for feature extraction",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./mvtec_anomaly_detection",
        help="Path to MVTec dataset",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Input image size (pixels)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for data loaders",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1,
        help="Batch size for data loaders",
    )
    parser.add_argument(
        "--max_train_images",
        type=int,
        default=60,
        help="Maximum number of training images to use (-1 for all)",
    )
    parser.add_argument(
        "--max_visualize_count",
        type=int,
        default=10,
        help="Maximum number of images to visualize during inference",
    )
    parser.add_argument(
        "--som_iterations",
        type=int,
        default=10,
        help="Number of iterations for SOM training",
    )
    parser.add_argument(
        "--som_input_dimensions",
        type=int,
        default=100,
        help="Number of feature dimensions to select for SOM input",
    )
    parser.add_argument(
        "--contour_threshold",
        type=float,
        default=0.6,
        help="Threshold for contour visualization",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.3,
        help="Learning rate for SOM training",
    )
    parser.add_argument(
        "--som_sigma",
        type=float,
        default=1.0,
        help="Initial neighborhood radius for SOM",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of nearest neurons to consider for anomaly scoring",
    )
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=4.0,
        help="Sigma for Gaussian blur applied to anomaly maps",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu')",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_argparse(args)

    print("Configuration:")
    for k, v in cfg.export_dict().items():
        print(f"  {k}: {v}")

    set_seed(cfg.seed)
    run_for_all_categories(cfg)


if __name__ == "__main__":
    main()
