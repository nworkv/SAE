# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
  def __init__(self, images_path, processor):
    self.processor = processor
    self.images = images_path
  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    image = Image.open(self.images[index]).convert("RGB")
    toks = self.processor(images=image, return_tensors="pt")
    return toks["pixel_values"].squeeze(0)