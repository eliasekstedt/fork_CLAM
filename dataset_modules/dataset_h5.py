
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
#import h5py

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self, fpath_qualityLog, wsi, img_transforms, fltr_params):
		self.map = self.apply_filter(fpath_qualityLog, fltr_params)
		self.patch_lvl = self.map['patch_lvl'].unique().item()
		self.patch_size = self.map['patch_size'].unique().item()
		self.wsi = wsi
		self.roi_transforms = img_transforms
		#self.file_path = fpath_wsiCoords
		#self.summary()

	def apply_filter(self, fpath_qualityLog, fltr_params):
		qlog = pd.read_csv(fpath_qualityLog)
		qlog = qlog[qlog['on_bg'] >= fltr_params['ll_bg']]
		qlog = qlog[qlog['on_blur'] >= fltr_params['ll_blur']]
		qlog = qlog[qlog['on_dist'] >= fltr_params['ll_dist']]
		qlog = qlog[qlog['on_dist'] <= fltr_params['ul_dist']]
		qlog = qlog.sample(frac=1) # important for random origin of patch on slide distribution into bags
		return qlog
			
	def __len__(self):
		return self.map.shape[0]

	def __getitem__(self, idx):
		row = self.map.iloc[idx]
		coord = np.array((row['pos_x'], row['pos_y']), dtype=np.int32)
		img = self.wsi.read_region(coord, self.patch_lvl, (self.patch_size, self.patch_size)).convert('RGB')
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}
	
	def summary(self):
		#hdf5_file = h5py.File(self.file_path, "r")
		#dset = hdf5_file['coords']
		#for name, value in dset.attrs.items():
		#	print(name, value)
		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

class Dataset_All_Bags(Dataset):
	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]





