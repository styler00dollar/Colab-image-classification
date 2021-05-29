import torch
import glob
from efficientnet_pytorch import EfficientNet
import cv2
import torch.nn.functional as F
import shutil
import os
from tqdm import tqdm
import numpy as np
import PIL
from PIL import Image

import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


def main():
    if cfg['model_train'] == 'efficientnet-b0':
      model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=cfg['num_classes'])
    elif cfg['model_train'] == 'efficientnet-b1':
      model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=cfg['num_classes'])
    elif cfg['model_train'] == 'efficientnet-b2':
      model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=cfg['num_classes'])
    elif cfg['model_train'] == 'efficientnet-b3':
      model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=cfg['num_classes'])
    elif cfg['model_train'] == 'efficientnet-b4':
      model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=cfg['num_classes'])
    elif cfg['model_train'] == 'efficientnet-b5':
      model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=cfg['num_classes'])
    elif cfg['model_train'] == 'efficientnet-b6':
      model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=cfg['num_classes'])
    elif cfg['model_train'] == 'efficientnet-b7':
      model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=cfg['num_classes'])

    model.load_state_dict(torch.load(cfg['model_path'], map_location=torch.device('cpu')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(cfg['path0']):
        os.makedirs(cfg['path0'])
    if not os.path.exists(cfg['path1']):
        os.makedirs(cfg['path1'])

    files = glob.glob(cfg['data_input_path'] + '/**/*.png', recursive=True)
    files_jpg = glob.glob(cfg['data_input_path'] + '/**/*.jpg', recursive=True)
    files.extend(files_jpg)

    if cfg['precision'] == 16:
      model.half()
    model.to(device)
    model.eval()

    with torch.no_grad():
      for f in tqdm(files):
          image = cv2.imread(f)
          #####################################
          if cfg['resize_method'] == "OpenCV":
              resized = cv2.resize(image, (cfg['size'],cfg['size']), interpolation=cv2.INTER_AREA)

          # resize with PIL
          elif cfg['resize_method'] == "PIL":
            image = Image.fromarray(image)
            image = image.resize((cfg['size'],cfg['size']))
            resized = np.asarray(image)
          #####################################

          image = torch.from_numpy(resized).unsqueeze(0).permute(0,3,1,2)/255

          image=image.to(device)
          if device == 'cuda' and cfg['precision'] == 16::
              image=image.type(torch.cuda.HalfTensor)

          y_pred= model(image)

          y_prob = torch.softmax(y_pred, dim=1)
          top_pred = y_prob.argmax(1, keepdim = True)


          if top_pred == 0:
            shutil.move(f, os.path.join(cfg['path0'], os.path.basename(f)))
          elif top_pred == 1:
            shutil.move(f, os.path.join(cfg['path1'], os.path.basename(f)))

if __name__ == "__main__":
    main()
