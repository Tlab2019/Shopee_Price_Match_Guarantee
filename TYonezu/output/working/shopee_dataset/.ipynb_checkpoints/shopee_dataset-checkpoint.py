import pandas as pd
import numpy as np


class ShopeeDataset(Dataset):
    def __init__(self, csv, transforms=None):

        self.csv = csv.reset_index()
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        text = row.title
        
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']       
        
        
        return image,torch.tensor(row.label_group)