
import torch
from torch.utils.data import Dataset

class MILReader(Dataset):
    def __init__(self, map, dpath_features_pt, mode):
        assert mode in ['train', 'test']
        self.mode = mode
        self.dpath_features_pt = dpath_features_pt
        self.map = map
    
    def __len__(self):
        return self.map.shape[0]
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        bag_id = row['bag_id']
        fpath_raw_features = self.dpath_features_pt / f"{bag_id}.pt"
        features = torch.load(fpath_raw_features)
        label = torch.tensor(row['label'].item(), dtype=torch.long)
        if self.mode == 'train':
            return features, label
        else:
            return bag_id, features, label