import os
import torch
import ast
import pandas as pd
from PIL import Image


class SpyFishAotearoaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        self.table = pd.read_csv(os.path.join(root_dir, "None_classifications.csv"), converters={
            "x": ast.literal_eval,
            "y": ast.literal_eval,
            "h": ast.literal_eval,
            "w": ast.literal_eval,
            "label": ast.literal_eval
        })
        self.imgs = os.path.join(root_dir, "Images")

    def __getitem__(self, idx):
        img_name = self.table.loc[idx].https_location.split('/')[-1]
        img_path = os.path.join(self.root_dir, "Images", img_name).replace("\\", "/")

        img = Image.open(img_path).convert("RGB")

        num_objs = len(self.table.loc[idx].x)
        boxes = []

        for i in range(num_objs):
            x = int(self.table.loc[idx].x[i])
            y = int(self.table.loc[idx].y[i])
            width = int(self.table.loc[idx].w[i])
            height = int(self.table.loc[idx].h[i])

            boxes.append([x, y, x + width, y + height])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        labels = torch.as_tensor(self.table.loc[idx].label, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.table)
