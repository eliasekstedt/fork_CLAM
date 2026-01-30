
import pandas as pd
from torch.utils.data import Dataset
import h5py

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self, file_path, wsi, img_transforms, quality_filter):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms
		self.file_path = file_path
		self.quality_filter = quality_filter

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		reject = self.quality_filter.apply(img)

		"""
		"""
		import numpy as np
		if reject and np.random.uniform(0, 1) < 1e-4:
			from pathlib import Path
			dpath_rejects = Path('_rejects')
			dpath_rejects.mkdir(exist_ok=True)
			fpath_reject = dpath_rejects / f"reject_{len(list(dpath_rejects.iterdir()))}.png"
			img.save(fpath_reject)

		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord, 'not_reject':not reject}

class Dataset_All_Bags(Dataset):
	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]





