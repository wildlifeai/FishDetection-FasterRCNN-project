import os
import torch
import pandas as pd
from PIL import Image


class SpyFishAotearoaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.table = pd.read_csv(os.path.join(root_dir, "None_classifications.csv"))
        self.imgs = os.path.join(root_dir, "Images")

    def __getitem__(self, idx):
        img_name = self.table[idx].https_location.split('/')[-1]
        img_path = os.path.join(self.root, "Images", img_name)

        img = Image.open(img_path).convert("RGB")

        num_objs = len(self.table[idx].x)
        boxes = []

        for i in range(num_objs):
            x = self.table[idx].x[i]
            y = self.table[idx].y[i]
            width = self.table[idx].w[i]
            height = self.table[idx].h[i]

            boxes.append([x, y, x + width, y + height])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        labels = self.table[idx].labels
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area}

        return img, target

    def __len__(self):
        return len(self.table)
