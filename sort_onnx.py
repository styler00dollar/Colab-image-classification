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
import onnx
import onnx_tensorrt.backend as backend

"""
def center_crop(img, dim):
          width, height = img.shape[1], img.shape[0]	#process crop width and height for max available dimension
          crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
          crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
          mid_x, mid_y = int(width/2), int(height/2)
          cw2, ch2 = int(crop_width/2), int(crop_height/2) 
          crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
          return crop_img
"""


def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)  # only difference


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_input_path", type=str, required=True, help="Input folder."
    )
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--path0", type=str, required=True)

    parser.add_argument("--path1", type=str, required=True)

    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--height_min", type=int, required=True)
    parser.add_argument("--width_min", type=int, required=True)

    # parser.add_argument('--fp16_mode', type=bool, default=False, required=False)
    args = parser.parse_args()

    model = onnx.load(args.onnx_path)
    # engine = backend.prepare(model, device='CUDA:0', int8_mode=True)
    engine = backend.prepare(model, device="CUDA:0", fp16_mode=True)

    if not os.path.exists(args.path0):
        os.makedirs(args.path0)
    if not os.path.exists(args.path1):
        os.makedirs(args.path1)

    files = glob.glob(args.data_input_path + "/**/*.png", recursive=True)
    files_jpg = glob.glob(args.data_input_path + "/**/*.jpg", recursive=True)
    files.extend(files_jpg)

    # height_min = 256
    # width_min = 256

    with torch.no_grad():
        for f in tqdm(files):
            image = cv2.imread(f)
            # image = cv2.resize(image, (256,256))

            # resizing to match original training, or detections will be bad
            height = image.shape[0]
            width = image.shape[1]
            if height > args.height_min and width > args.width_min:
                height_resized = args.height_min
                if width < height:
                    scale_x = args.width_min / width
                    width_resized = args.width_min
                    height_resized = scale_x * height
                else:
                    scale_y = args.height_min / height
                    height_resized = args.height_min
                    width_resized = scale_y * width
                    image = cv2.resize(
                        image, (round(width_resized), round(height_resized))
                    )
            elif height <= args.height_min or width <= args.width_min:
                if height > width:
                    width_resized = args.width_min
                    scale = args.width_min / width
                    height_resized = height * scale
                    image = cv2.resize(
                        image, (round(width_resized), round(height_resized))
                    )
                else:
                    height_resized = args.height_min
                    scale = args.height_min / height
                    width_resized = width * scale
                    image = cv2.resize(
                        image, (round(width_resized), round(height_resized))
                    )

            image = crop_center(image, args.width_min, args.height_min)
            image = np.swapaxes(image, 0, 2)
            image = np.expand_dims(image, axis=0)
            image = image.astype(dtype=np.float32)

            y_pred = engine.run(image)[0]
            # y_pred = engine.run(image)

            y_pred = softmax(y_pred)
            top_pred = np.argmax(y_pred, axis=1).astype(np.uint8)

            print(f)
            print(top_pred)
            print("####################")
            # if top_pred == 1:
            #  shutil.move(f, os.path.join(args.path0, os.path.basename(f)))
            # elif top_pred == 2:
            #  shutil.move(f, os.path.join(args.path1, os.path.basename(f)))


if __name__ == "__main__":
    main()
