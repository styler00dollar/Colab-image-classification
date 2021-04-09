import argparse
from torch.utils.data import Dataset
import os
import cv2
import torch
import glob
from CustomTrainClass import CustomTrainClass
import pytorch_lightning as pl
import numpy as np
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
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_input_path', type=str, required=True, help='Input folder.')
    parser.add_argument('--netG_pth_path', type=str, required=True, help='Model path.')
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--model_train', type=str, required=True)

    #parser.add_argument('--fp16_mode', type=bool, default=False, required=False)
    args = parser.parse_args()

    dm = DS_val(root = args.data_input_path)

    model = CustomTrainClass(model_train=args.model_train, num_classes=args.num_classes, diffaug_activate=False, policy=None)
    # skipping validation with limit_val_batches=0
    #gpus=1, limit_val_batches=0,
    trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=20)

    model.netD.load_state_dict(torch.load(args.netG_pth_path))

    trainer.test(model, dm)

if __name__ == "__main__":
    main()
