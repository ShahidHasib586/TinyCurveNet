import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def list_images(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
    files.sort()
    return files

class PairedImageFolder(Dataset):
    def __init__(self, low_dir, high_dir, img_size=256, train=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.files = list_images(low_dir)
        self.files = [f for f in self.files if os.path.exists(os.path.join(high_dir, f))]
        if len(self.files) == 0:
            raise RuntimeError(f"No paired images found in {low_dir} and {high_dir}")

        if train:
            self.tf = T.Compose([
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        low = Image.open(os.path.join(self.low_dir, name)).convert("RGB")
        high = Image.open(os.path.join(self.high_dir, name)).convert("RGB")
        low = self.tf(low)
        high = self.tf(high)
        return low, high, name
