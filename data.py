import os
import glob
import cv2
import yaml
from torch.utils.data import Dataset

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

if cfg["img_reader"] == "turboJPEG":
    from turbojpeg import TurboJPEG

    jpeg_reader = TurboJPEG()


class ImageDataloader(Dataset):
    def __init__(self, data_root, size, means, std, transform=None, ffcv=False):
        self.img_paths = []
        self.labels = []
        self.size = size
        self.transform = transform
        self.load_data(data_root)
        self.ffcv = ffcv

    def load_data(self, root):
        # Loop through each subfolder in sorted order
        class_dirs = sorted(os.listdir(root))
        for label, class_dir in enumerate(class_dirs):
            class_path = os.path.join(root, class_dir)
            if os.path.isdir(class_path):
                image_files = glob.glob(os.path.join(class_path, "*"))
                # Filter out non-image files, if any
                image_files = [
                    f
                    for f in image_files
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
                ]
                # Append image paths and their corresponding class labels based on folder index
                self.img_paths.extend(image_files)
                self.labels.extend([label] * len(image_files))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        if cfg["img_reader"] == "OpenCV":
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif cfg["img_reader"] == "turboJPEG":
            img = jpeg_reader.decode(open(img_path, "rb").read(), 0)  # 0 = RGB

        if not self.ffcv:
            img = self.transform(img)
        img_class = self.labels[index]
        return img, img_class
