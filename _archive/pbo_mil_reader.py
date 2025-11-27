
import os
import pandas as pd
from torch.utils.data import Dataset
import torch


class Reader(Dataset):
    def __init__(self, fpath_map, dpath_features_pt, mode):
        assert mode in ['train', 'test']
        self.dpath_features_pt = dpath_features_pt
        self.map = pd.read_csv(fpath_map)
    
    def __len__(self):
        return self.map.shape[0]
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        slide_id = row['slide_id']
        fpath_raw_features = os.path.join(self.dpath_features_pt, f"{slide_id}.pt")
        features = torch.load(fpath_raw_features)
        label = torch.tensor([row['label'].item()], dtype=torch.float32)
        if self.mode == 'train':
            return slide_id, features
        else:
            return slide_id, features, label


