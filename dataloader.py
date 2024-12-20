from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision.datasets as datasets
import yaml
from data import ImageDataloader

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        training_path: str = "./",
        validation_path: str = "./",
        test_path: str = "./",
        batch_size: int = 5,
        num_workers: int = 2,
        size=256,
        means=[0.7000, 0.6413, 0.6352],
        std=[0.2529, 0.2519, 0.2450],
    ):
        super().__init__()
        self.training_dir = training_path
        self.validation_dir = validation_path
        self.test_dir = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size

        self.means = means
        self.std = std

    def setup(self, stage=None):
        img_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(self.size, self.size)),
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=self.means, std=self.std),
            ]
        )

        img_transforms_val = transforms.Compose(
            [
                transforms.Resize(size=(self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.means, std=self.std),
            ]
        )

        self.DS_train = ImageDataloader(
            data_root=self.training_dir,
            size=self.size,
            means=self.means,
            std=self.std,
            transform=img_transforms,
        )
        self.DS_validation = datasets.ImageFolder(
            root=self.validation_dir, transform=img_transforms_val
        )
        self.DS_test = datasets.ImageFolder(
            root=self.test_dir, transform=img_transforms_val
        )

    def train_dataloader(self):
        return DataLoader(
            self.DS_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.DS_validation, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.DS_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
