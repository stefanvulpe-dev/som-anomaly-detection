import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    backbone: str
    data_dir: str
    out_dir: str
    img_size: int
    train_batch_size: int
    test_batch_size: int
    max_train_images: int
    max_visualize_count: int
    som_iterations: int
    som_input_dimensions: int
    contour_threshold: float
    learning_rate: float
    som_sigma: float
    k: int
    blur_sigma: float
    seed: int
    num_workers: int
    device: str

    def from_argparse(args: argparse.Namespace) -> "Config":
        return Config(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            img_size=args.img_size,
            backbone=args.backbone,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            max_train_images=args.max_train_images,
            max_visualize_count=args.max_visualize_count,
            som_iterations=args.som_iterations,
            som_input_dimensions=args.som_input_dimensions,
            contour_threshold=args.contour_threshold,
            learning_rate=args.learning_rate,
            som_sigma=args.som_sigma,
            k=args.k,
            blur_sigma=args.blur_sigma,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
        )

    def export_dict(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "out_dir": self.out_dir,
            "img_size": self.img_size,
            "backbone": self.backbone,
            "train_batch_size": self.train_batch_size,
            "test_batch_size": self.test_batch_size,
            "max_train_images": self.max_train_images,
            "max_visualize_count": self.max_visualize_count,
            "som_iterations": self.som_iterations,
            "som_input_dimensions": self.som_input_dimensions,
            "contour_threshold": self.contour_threshold,
            "learning_rate": self.learning_rate,
            "som_sigma": self.som_sigma,
            "k": self.k,
            "blur_sigma": self.blur_sigma,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "device": self.device,
        }
