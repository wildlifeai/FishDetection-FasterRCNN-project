import os
import torch
import ast
import pandas as pd
from PIL import Image


class SpyFishAotearoaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, file_name, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        file_path = os.path.join(root_dir, "output", file_name)
        self.table = pd.read_csv(file_path, converters={
            "x1": ast.literal_eval,
            "y1": ast.literal_eval,
            "x2": ast.literal_eval,
            "y2": ast.literal_eval,
            "label": ast.literal_eval
        })
        self.imgs_path = os.path.join(root_dir, "images")

    def get_image_name(self, idx):
        return self.table.loc[idx].image_name

    def __getitem__(self, idx):
        img_name = self.table.loc[idx].image_name
        img_path = os.path.join(self.imgs_path, img_name)

        img = Image.open(img_path)

        num_objs = len(self.table.loc[idx].x1)
        boxes = []

        for i in range(num_objs):
            x1 = self.table.loc[idx].x1[i]
            y1 = self.table.loc[idx].y1[i]
            x2 = self.table.loc[idx].x2[i]
            y2 = self.table.loc[idx].y2[i]

            boxes.append([x1, y1, x2, y2])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        labels = torch.as_tensor(self.table.loc[idx].label, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def __len__(self):
        return len(self.table)
