from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision.datasets as datasets
from RandAugment import RandAugment
import yaml

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
        img_tf = transforms.Compose(
            [
                transforms.Resize(size=(self.size, self.size)),
                transforms.RandomRotation(5),
                transforms.CenterCrop(size=self.size),
                # transforms.RandomCrop(self.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.means, std=self.std),
            ]
        )

        img_tf_val = transforms.Compose(
            [
                transforms.Resize(size=(self.size, self.size)),
                # transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.means, std=self.std),
            ]
        )

        if cfg["aug"] == "RandAugment":
            img_tf.transforms.insert(0, RandAugment(2, 2))

        self.DS_train = datasets.ImageFolder(root=self.training_dir, transform=img_tf)
        self.DS_validation = datasets.ImageFolder(
            root=self.validation_dir, transform=img_tf_val
        )
        self.DS_test = datasets.ImageFolder(root=self.test_dir, transform=img_tf_val)

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
