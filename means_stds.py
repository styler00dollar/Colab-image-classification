import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse
def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True, help='Input folder.')
    #parser.add_argument('--fp16_mode', type=bool, default=False, required=False)
    args = parser.parse_args()


    train_data = datasets.ImageFolder(root = args.train_dir,
                                      transform = transforms.ToTensor())

    means = torch.zeros(3)
    stds = torch.zeros(3)

    for img, label in tqdm(train_data):
        means += torch.mean(img, dim = (1,2))
        stds += torch.std(img, dim = (1,2))

    means /= len(train_data)
    stds /= len(train_data)

    print("\n")
    print(f'Calculated means: {means}')
    print(f'Calculated stds: {stds}')

if __name__ == "__main__":
    main()
