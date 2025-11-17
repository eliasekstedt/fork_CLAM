
import os
import torch
import h5py
import openslide
#from PIL import Image
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader, Dataset
from pbo.pbo_config import cfg
from tqdm import tqdm
import matplotlib.pyplot as plt

def cshow(img):
    plt.imshow(img)
    plt.show()

def conclude_measurements(measurements):
    assem = None
    for measurement in measurements:
        measurement = measurement.unsqueeze(0)
        if assem is None:
            assem = measurement
        else:
            assem = torch.cat([assem, measurement], axis=0)
    return torch.mean(assem, axis=0)

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
for fname in tqdm(fname_patchset):
    fname_slide = f"{fname.rstrip('.h5')}.mrxs"
    fpath_patchset = os.path.join(dpath_patchset, fname)
    fpath_slide = os.path.join(dpath_mrxsRoot, fname_slide)

    wsi = openslide.open_slide(fpath_slide)
    reader = WSI4NormReader(fpath_patchset, wsi)
    loader = DataLoader(reader, batch_size=32, shuffle=True)

    patch_means, patch_stds = [], []
    for i, item in enumerate(loader):
        if i > 0:
            break
        img = item['img'][0, :, :, :]
        patch_means.append(torch.mean(img, axis=(1, 2)))
        patch_stds.append(torch.std(img, axis=(1, 2)))
        #cshow(img.permute(1, 2, 0))

	
mean = conclude_measurements(patch_means)
std = conclude_measurements(patch_stds)
print(f"mean: {mean}\nstd: {std}")
