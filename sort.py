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
import timm
import torchvision

model_path = ""
model_choise = ""
size = 256
half = False
resize_method = "OpenCV" # OpenCV | PIL

input_path = "/media/"
path0 = "0"
path1 = "1"

def main():
    model = timm.create_model(model_choise, num_classes=2, pretrained=True)
    #from efficientnet_pytorch import EfficientNet
    #model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(path0):
        os.makedirs(path0)
    if not os.path.exists(path1):
        os.makedirs(path1)

    files = glob.glob(input_path + '/**/*.png', recursive=True)
    files_jpg = glob.glob(input_path + '/**/*.jpg', recursive=True)
    files.extend(files_jpg)

    if half == True:
      model.half()
    model.to(device)
    model.eval()

    with torch.no_grad():
      for f in tqdm(files):
          image = cv2.imread(f)
          #####################################
          if resize_method == "OpenCV":
              resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

          # resize with PIL
          elif resize_method == "PIL":
            image = Image.fromarray(image)
            image = image.resize((size, size))
            resized = np.asarray(image)
          #####################################

          image = torch.from_numpy(resized).unsqueeze(0).permute(0,3,1,2)/255

          image = torchvision.transforms.functional.normalize(image, mean=[0.7032, 0.6346, 0.6234], std=[0.2520, 0.2507, 0.2417])

          image=image.to(device)
          if half == True:
              image=image.type(torch.cuda.HalfTensor)

          y_pred= model(image)

          y_prob = torch.softmax(y_pred, dim=1)
          top_pred = y_prob.argmax(1, keepdim = True)


          """
          if top_pred == 0:
            shutil.move(f, os.path.join(path0, os.path.basename(f)))
          elif top_pred == 1:
            shutil.move(f, os.path.join(path1, os.path.basename(f)))
          """

          if top_pred == 1 and y_prob[0][1] > 0.98:
            shutil.copy(f, os.path.join(path1, os.path.basename(f)))

if __name__ == "__main__":
    main()
