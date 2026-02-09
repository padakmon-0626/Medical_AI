import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.data_dir = os.path.join(root_dir, split)
        self.normal_images = glob.glob(os.path.join(self.data_dir, "normal", "*.jpg"))
        self.tumor_images = glob.glob(os.path.join(self.data_dir, "tumor", "*.jpg"))
        self.all_images = self.normal_images + self.tumor_images
        self.labels = [0] * len(self.normal_images) + [1] * len(self.tumor_images)
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, img_path