
import os
import torch
import h5py
import openslide
#from PIL import Image
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader, Dataset
from pbo_config import cfg
from tqdm import tqdm
import matplotlib.pyplot as plt

def cshow(img):
    plt.imshow(img)
    plt.show()

def assemble_log(measurements):
    log = None
    for measurement in measurements:
        measurement = measurement.unsqueeze(0)
        if log is None:
            log = measurement
        else:
            log = torch.cat([log, measurement], axis=0)
    return log

def get_result(log):
    return torch.mean(log, axis=0)

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

import numpy as np
n = 1
lim = np.inf

assert lim > 0 and n >= 0
dpath_patchset = cfg.dpath_patchset
dpath_mrxsRoot = cfg.dpath_mrxsRoot

fname_patchset = os.listdir(dpath_patchset)

log_patchset_means, log_patchset_stds = [], []
for _ in tqdm(range(n), total=n):
#for _ in tqdm(range(n), total=n):
    for fname in tqdm(fname_patchset, total=len(fname_patchset)):
    #for fname in fname_patchset:
        fname_slide = f"{fname.rstrip('.h5')}.mrxs"
        fpath_patchset = os.path.join(dpath_patchset, fname)
        fpath_slide = os.path.join(dpath_mrxsRoot, fname_slide)

        wsi = openslide.open_slide(fpath_slide)
        reader = WSI4NormReader(fpath_patchset, wsi)
        loader = DataLoader(reader, batch_size=32, shuffle=True)
        print('\n', len(loader))
        raise SystemExit

        patch_means, patch_stds = [], []
        for item in tqdm(enumerate(loader), total=len(loader)):
            img = item['img'][0, :, :, :]
            patch_means.append(torch.mean(img, axis=(1, 2)))
            patch_stds.append(torch.std(img, axis=(1, 2)))
            #cshow(img.permute(1, 2, 0))
	
    patch_means_log = assemble_log(patch_means)
    patch_stds_log = assemble_log(patch_stds)
    patchset_mean = get_result(patch_means_log)
    patchset_std = get_result(patch_stds_log)
    log_patchset_means.append(patchset_mean)
    log_patchset_stds.append(patchset_std)
    print(f"mean: {patchset_mean}\nstd: {patchset_std}")

import pandas as pd
log_patchset_means = pd.DataFrame(assemble_log(log_patchset_means).detach().cpu().numpy())
log_patchset_means.columns = ['mu_g', 'mu_r', 'mu_b']
log_patchset_stds = pd.DataFrame(assemble_log(log_patchset_stds).detach().cpu().numpy())
log_patchset_stds.columns = ['std_g', 'std_r', 'std_b']
full_log = pd.concat([log_patchset_means, log_patchset_stds], axis=1)
full_log['lim'] = lim

if os.path.isfile('trfs_log.csv'):
    full_log = pd.concat([pd.read_csv('trfs_log.csv'), full_log], axis=0)


print(full_log)
full_log.to_csv('trfs_log.csv', index=False)
