import torch
from typing import Iterable
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms


SEGMENT_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                  [255, 0, 85], [255, 0, 170],
                  [0, 255, 0], [85, 255, 0], [170, 255, 0],
                  [0, 255, 85], [0, 255, 170],
                  [0, 0, 255], [85, 0, 255], [170, 0, 255],
                  [0, 85, 255], [0, 170, 255],
                  [255, 255, 0], [255, 255, 85], [255, 255, 170],
                  [255, 0, 255], [255, 85, 255], [255, 170, 255],
                  [0, 255, 255], [85, 255, 255], [170, 255, 255]]

SEGCOL_DICT_BGR = {"skin": ((255, 0, 0),),  # ?
                   "face": ((255, 85, 0),),
                   "brows": ((255, 0, 85), (255, 170, 0)),
                   "eyes": ((255, 0, 170), (0, 255, 0)),
                   "eye_glasses": ((85, 255, 0),),  # ?
                   "ears": ((0, 255, 85), (170, 255, 0)),
                   "nose": ((0, 0, 255),),
                   "mouth_all": ((85, 0, 255), (170, 0, 255), (0, 85, 255)),
                   "neck": ((0, 170, 255),),
                   "neck_l": ((255, 255, 0),),  # ?
                   "hair": ((255, 255, 170),),
                   "clothes": ((255, 255, 85),),
                   "hat": ((255, 0, 255),)}
SEGCOL_DICT_RGB = {"skin": ((0, 0, 255),),  # ?
                   "face": ((0, 85, 255),),
                   "brows": ((85, 0, 255), (0, 170, 255)),
                   "eyes": ((170, 0, 255), (0, 255, 0)),
                   "eye_glasses": ((0, 255, 85),),  # ?
                   "ears": ((85, 255, 0), (0, 255, 170)),
                   "nose": ((255, 0, 0),),
                   "mouth_all": ((255, 0, 85), (255, 0, 170), (255, 85, 0)),
                   "neck": ((255, 170, 0),),
                   "neck_l": ((0, 255, 255),),  # ?
                   "hair": ((170, 255, 255),),
                   "clothes": ((85, 255, 255),),
                   "hat": ((255, 0, 255),)}
K_DICT = {"skin": 1, "face": 2, "brows": 2, "eyes": 2, "eye_glasses": 2, "ears": 2, "nose": 2, "mouth_all": 3,
          "neck": 2, "neck_l": 2, "hair": 2, "clothes": 2, "hat": 2}


_standard_transform = transforms.Compose([
    transforms.ToTensor()
])


def transform_image(img: Image.Image, transform_type: str, background: np.ndarray = None, mask: np.ndarray = None):
    if background is not None and mask is not None:
        # add background from gt_image
            # make it numpy array and cut masked part
        img_bg = np.asarray(img)
        mask = mask.astype(np.uint8)
        img_bg = cv2.bitwise_and(img_bg, img_bg, mask=mask)
            # add background
        img = cv2.cvtColor(background, cv2.COLOR_BGR2RGB) + img_bg
            # convert back to PIL
        img = Image.fromarray(img)
        img.save("/disk/sdb1/avatars/sveta/TecoGAN_results/Multimodal/temp/patches/temp.jpg")
        #print("added background")

    if transform_type == 'standard':
        return standard_transform(img)
    if transform_type == 'semantic':
        return semantic_transform(img)
    if transform_type == 'semantic2':
        return semantic_transform2(img)
    if transform_type == 'normal_map':
        return normal_map_transform(img)
    raise NotImplementedError


def standard_transform(img: Image.Image):
    return _standard_transform(img)


def semantic_transform(img: Image.Image):
    img_array = np.array(img)
    h, w, _ = img_array.shape
    segments = np.zeros(shape=(len(SEGMENT_COLORS), h, w), dtype=np.uint8)
    for i, color in enumerate(SEGMENT_COLORS):
        segments[i, :, :] = np.all(img_array == color, axis=-1).astype(np.uint8)
    return torch.tensor(segments)

def semantic_transform2(img: Image.Image, categories: Iterable[str] = K_DICT):
    img_array = np.array(img)
    h, w, _ = img_array.shape
    segments = np.zeros(shape=(len(categories), h, w), dtype=np.uint8)
    for i, cat in enumerate(categories):
        colors = SEGCOL_DICT_RGB[cat]
        for color in colors:
            segments[i, :, :] += np.all(img_array == color, axis=-1).astype(np.uint8)
    return torch.tensor(segments)


def normal_map_transform(img: Image.Image):
    return _standard_transform(img)
