import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))

class PairedImageFolder(Dataset):
    """Aligned RGB pairs with shared random crop/flip; deterministic validation resize."""
    def __init__(self, low_dir, high_dir, img_size=256, train=True):
        self.low_dir, self.high_dir = low_dir, high_dir
        self.img_size, self.train = int(img_size), bool(train)
        self.files = [f for f in list_images(low_dir)
                      if os.path.exists(os.path.join(high_dir, f))]
        if not self.files:
            raise RuntimeError(f"No paired images found in {low_dir} and {high_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        with Image.open(os.path.join(self.low_dir, name)) as image:
            low = image.convert("RGB")
        with Image.open(os.path.join(self.high_dir, name)) as image:
            high = image.convert("RGB")
        if low.size != high.size:
            raise ValueError(f"Pair has different dimensions: {name}")
        if self.train:
            w, h = low.size
            padding = [0, 0, max(0, self.img_size-w), max(0, self.img_size-h)]
            if any(padding):
                low = TF.pad(low, padding, padding_mode="edge")
                high = TF.pad(high, padding, padding_mode="edge")
            crop = RandomCrop.get_params(low, (self.img_size, self.img_size))
            low, high = TF.crop(low, *crop), TF.crop(high, *crop)
            if torch.rand(()) < 0.5:
                low, high = TF.hflip(low), TF.hflip(high)
        else:
            low = TF.resize(low, [self.img_size, self.img_size])
            high = TF.resize(high, [self.img_size, self.img_size])
        return TF.to_tensor(low), TF.to_tensor(high), name
