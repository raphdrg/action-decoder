import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml


class CelebADataset(Dataset):
    def __init__(self, img_dir: str, image_size: int):
        self.img_dir = img_dir
        self.filenames = sorted([
            f for f in os.listdir(img_dir) if f.endswith(".jpg")
        ])
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),             # [0, 1]
            transforms.Normalize([0.5], [0.5]) # [-1, 1]
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.filenames[idx])
        img = Image.open(path).convert("RGB")
        return self.transform(img)


def get_dataloader(config_path: str = "config.yaml") -> DataLoader:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = CelebADataset(
        img_dir=os.path.join("data", "img_align_celeba"),
        image_size=cfg["image_size"],
    )
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
