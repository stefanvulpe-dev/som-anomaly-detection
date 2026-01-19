from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)

from src.config import Config


class MVTecDataset(Dataset):
    def __init__(
        self,
        root: str,
        category: str,
        split: str,
        transform: Compose = None,
        mask_transform: Compose = None,
    ):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform

        split_dir = self.root / category / split
        gt_dir = self.root / category / "ground_truth"

        self.samples = []
        for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
            for img_path in sorted(defect_dir.glob("*.png")):
                mask_path: Optional[Path]
                is_anomaly: int

                if split == "train":
                    mask_path = None
                    is_anomaly = 0
                else:
                    if defect_type == "good":
                        mask_path = None
                        is_anomaly = 0
                    else:
                        mask_path = (
                            gt_dir / defect_type / f"{img_path.stem}_mask.png"
                        )
                        if not mask_path.exists():
                            raise FileNotFoundError(
                                f"Missing mask for {img_path}: {mask_path}"
                            )
                        is_anomaly = 1

                self.samples.append(
                    (img_path, mask_path, defect_type, is_anomaly)
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        img_path, mask_path, defect_type, is_anomaly = self.samples[idx]

        image = default_loader(str(img_path))

        mask: Optional[Image.Image] = None
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        if mask is not None and self.mask_transform is not None:
            mask = self.mask_transform(mask)

        if mask is None:
            if isinstance(image, torch.Tensor):
                _, height, width = image.shape
                mask_tensor = torch.zeros(
                    (1, height, width), dtype=torch.float32
                )
            else:
                raise TypeError(
                    "Expected image to be a torch.Tensor after transform; "
                    "please provide a transform that ends with ToTensor()."
                )
        else:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(
                    "Expected mask to be a torch.Tensor after mask_transform; "
                    "please provide a mask_transform that ends with ToTensor()."
                )
            mask_tensor = mask

        if not isinstance(image, torch.Tensor):
            raise TypeError(
                "Expected image to be a torch.Tensor after transform; "
                "please provide a transform that ends with ToTensor()."
            )

        return image, mask_tensor, is_anomaly, defect_type, str(img_path)


def make_dataloaders(
    cfg: Config,
    category: str,
    image_transform: Compose,
    split: str,
) -> Tuple[DataLoader, MVTecDataset]:
    mask_transform = Compose(
        [
            Resize((cfg.img_size, cfg.img_size)),
            ToTensor(),
        ]
    )

    dataset = MVTecDataset(
        root=cfg.data_dir,
        category=category,
        split=split,
        transform=image_transform,
        mask_transform=mask_transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size
        if split == "train"
        else cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return dataloader, dataset
