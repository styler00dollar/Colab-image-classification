import torch
import glob
from efficientnet_pytorch import EfficientNet
import cv2
import torch.nn.functional as F
import shutil
import os
from tqdm import tqdm
import numpy as np
import argparse
import PIL
from PIL import Image

resize_method = 'PIL' #@param ["OpenCV", "PIL"] {allow-input: false}


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_input_path', type=str, required=True, help='Input folder.')
    parser.add_argument('--model_train', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--path0', type=str, required=True)

    parser.add_argument('--path1', type=str, required=True)

    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--height_min', type=int, required=True)
    parser.add_argument('--width_min', type=int, required=True)

    #parser.add_argument('--fp16_mode', type=bool, default=False, required=False)
    args = parser.parse_args()

    if args.model_train == 'efficientnet-b0':
      model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b1':
      model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b2':
      model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b3':
      model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b4':
      model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b5':
      model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b6':
      model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b7':
      model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=args.num_classes)

    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.path0):
        os.makedirs(args.path0)
    if not os.path.exists(args.path1):
        os.makedirs(args.path1)

    files = glob.glob(args.data_input_path + '/**/*.png', recursive=True)
    files_jpg = glob.glob(args.data_input_path + '/**/*.jpg', recursive=True)
    files.extend(files_jpg)

    #model.half()
    model.to(device)
    model.eval()

    #height_min = 256
    #width_min = 256

    with torch.no_grad():
      for f in tqdm(files):
          image = cv2.imread(f)
          """
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          # resizing to match original training, or detections will be bad
          height = image.shape[0]
          width = image.shape[1]
          if height > args.height_min and width > args.width_min:
              height_resized = args.height_min
              if width < height:
                scale_x = args.width_min/width
                width_resized = args.width_min
                height_resized = scale_x * height
              else:
                scale_y = args.height_min/height
                height_resized = args.height_min
                width_resized = scale_y * width
                image = cv2.resize(image, (round(width_resized), round(height_resized)))
          elif height <= args.height_min or width <= args.width_min:
              if height > width:
                  width_resized = args.width_min
                  scale = args.width_min/width
                  height_resized = height*scale
                  image = cv2.resize(image, (round(width_resized), round(height_resized)))
              else:
                  height_resized = args.height_min
                  scale = args.height_min/height
                  width_resized = width*scale
                  image = cv2.resize(image, (round(width_resized), round(height_resized)))
          """
          #####################################
          if resize_method == "OpenCV":
              resized = cv2.resize(image, (args.height_min,args.width_min), interpolation=cv2.INTER_AREA)

          # resize with PIL
          elif resize_method == "PIL":
            image = Image.fromarray(image)
            image = image.resize((args.height_min,args.width_min))
            resized = np.asarray(image)
          #####################################

          image = torch.from_numpy(resized).unsqueeze(0).permute(0,3,1,2)/255

          image=image.to(device)
          if device == 'cuda':
              image=image.type(torch.cuda.HalfTensor)

          y_pred= model(image)

          y_prob = torch.softmax(y_pred, dim=1)
          top_pred = y_prob.argmax(1, keepdim = True)


          if top_pred == 0:
            shutil.move(f, os.path.join(args.path0, os.path.basename(f)))
          elif top_pred == 1:
            shutil.move(f, os.path.join(args.path1, os.path.basename(f)))

if __name__ == "__main__":
    main()
