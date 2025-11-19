
import os
import torch
import h5py
import openslide
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader, Dataset
from pbo.pbo_config import cfg
from tqdm import tqdm


class WSI4NormReader(Dataset):
	def __init__(self, file_path, wsi):
		self.file_path = file_path
		self.wsi = wsi

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		img = Compose([ToTensor()])(img)
		return {'img': img, 'coord': coord}


dpath_patchset = cfg.dpath_patchset
dpath_mrxsRoot = cfg.dpath_mrxsRoot

fname_patchset = os.listdir(dpath_patchset)
patchset_stats = None
for fname in tqdm(fname_patchset, total=len(fname_patchset)):
    fname_slide = f"{fname.rstrip('.h5')}.mrxs"
    fpath_patchset = os.path.join(dpath_patchset, fname)
    fpath_slide = os.path.join(dpath_mrxsRoot, fname_slide)

    wsi = openslide.open_slide(fpath_slide)
    reader = WSI4NormReader(fpath_patchset, wsi)
    loader = DataLoader(reader, batch_size=512, shuffle=False)

    batch_statitem = None
    for item in loader:
        img = item['img'].to('cuda:0')
        new_statitem = torch.concat([
              torch.mean(img, axis=(0, 2, 3)).unsqueeze(0),
              torch.std(img, axis=(0, 2, 3)).unsqueeze(0)
        ], axis=1)
        if batch_statitem is None:
            batch_statitem = new_statitem
        else:
            batch_statitem = torch.concat([
                batch_statitem,
                new_statitem,
            ], axis=0)

    batch_statitem = torch.mean(batch_statitem, axis=0).unsqueeze(0)

    if patchset_stats is None:
        patchset_stats = batch_statitem
    else:
        patchset_stats = torch.cat([
            patchset_stats,
            batch_statitem,
        ], axis=0)

        


import pandas as pd
patchset_stats = pd.DataFrame(patchset_stats.cpu().numpy())
patchset_stats.columns = ['mu_g', 'mu_r', 'mu_b', 'sd_g', 'sd_r', 'sd_b']
print(patchset_stats)
print(patchset_stats.mean())
patchset_stats.to_csv('trfs_log.csv', index=False)


