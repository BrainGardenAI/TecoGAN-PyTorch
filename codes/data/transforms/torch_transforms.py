import torch
from PIL import Image
import numpy as np
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


_standard_transform = transforms.Compose([
    transforms.ToTensor()
])


def transform_image(img: Image.Image, transform_type: str):
    if transform_type == 'standard':
        return standard_transform(img)
    if transform_type == 'semantic':
        return semantic_transform(img)
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


def normal_map_transform(img: Image.Image):
    return _standard_transform(img)
