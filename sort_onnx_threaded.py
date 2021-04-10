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

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_input_path', type=str, required=True, help='Input folder.')
    parser.add_argument('--model_train', type=str, required=True)
    parser.add_argument('--onnx_path', type=str, required=True)
    parser.add_argument('--path0', type=str, required=True)

    parser.add_argument('--path1', type=str, required=True)

    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--height_min', type=int, required=True)
    parser.add_argument('--width_min', type=int, required=True)

    #parser.add_argument('--fp16_mode', type=bool, default=False, required=False)
    args = parser.parse_args()

    model = onnx.load(args.onnx_path)
    engine = backend.prepare(model, device='CUDA:0', int8_mode=True)


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

    model.to(device)



def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
        for i in range(wanted_parts) ]

files_splitted = split_list(file_list, wanted_parts=2) #16 mit 16 threads

def do_1():
    with torch.no_grad():
      for f in tqdm(files_splitted[0]):
          image = cv2.imread(f)
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

          image = torch.from_numpy(image).unsqueeze(0).permute(0,3,1,2)/255

          image=image.to(device)

          y_pred = engine.run(image)[0]

          y_pred = softmax(y_pred)
          top_pred = np.argmax(y_pred, axis=1).astype(np.uint8)

          if top_pred == 2:
            shutil.copy(f, os.path.join(args.path1, os.path.basename(f)))

def do_2():
    with torch.no_grad():
      for f in files_splitted[1]:
          image = cv2.imread(f)
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

          image = torch.from_numpy(image).unsqueeze(0).permute(0,3,1,2)/255

          image=image.to(device)

          y_pred = engine.run(image)[0]

          y_pred = softmax(y_pred)
          top_pred = np.argmax(y_pred, axis=1).astype(np.uint8)

          if top_pred == 2:
            shutil.copy(f, os.path.join(args.path1, os.path.basename(f)))

if __name__ == "__main__":
    # create threads
    main()
    t1 = threading.Thread(target=do_1)
    t2 = threading.Thread(target=do_2)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Done!")
