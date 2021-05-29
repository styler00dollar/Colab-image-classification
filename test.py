from torch.utils.data import Dataset
import os
import cv2
import torch
import glob
from CustomTrainClass import CustomTrainClass
import pytorch_lightning as pl
import numpy as np

import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

class DS_val(Dataset):
    def __init__(self, root):
        self.samples = []
        self.samples = glob.glob(root + '/**/*.png', recursive=True)
        self.samples_jpg = glob.glob(root + '/**/*.jpg', recursive=True)
        self.samples.extend(self.samples_jpg)

        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        sample = cv2.imread(sample_path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        sample = torch.from_numpy(sample.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)/255
        return sample, sample_path

def main():
    dm = DS_val(root = cfg['test_path'])

    model = CustomTrainClass(model_train=cfg['model_train'], num_classes=cfg['num_classes'], diffaug_activate=cfg['diffaug_activate'], policy=cfg['policy'])
    # skipping validation with limit_val_batches=0
    #gpus=1, limit_val_batches=0,
    trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=20)

    model.netD.load_state_dict(torch.load(cfg['model_path']))

    trainer.test(model, dm)

if __name__ == "__main__":
    main()
