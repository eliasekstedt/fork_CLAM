# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
from tqdm import tqdm
#import time
#import pandas as pd
from PIL import Image

def dicprint(dict, tag):
	print(tag)
	for key in dict.keys():
		print(key, dict[key])
	print('\n')

class PatchsetGenerator:
	def __init__(self, dpath_mrxs, dpath_patchRoot, dpath_patchset, dpath_patchset_masks,
		dpath_patchset_stitch, param_sthresh, param_mthresh, param_close,
		param_otsu,
	):
		self.dpath_mrxs = dpath_mrxs
		self.dpath_patchRoot = dpath_patchRoot
		self.dpath_patchset = dpath_patchset
		self.dpath_patchset_masks = dpath_patchset_masks
		self.dpath_patchset_stitch = dpath_patchset_stitch
		self.patch_size = 256
		self.step_size = 256
		self.seg_params = {
			'seg_level': -1,
			'sthresh': param_sthresh,
			'mthresh': param_mthresh,
			'close': param_close,
			'use_otsu': param_otsu,
			'keep_ids': 'none',
			'exclude_ids': 'none',
		}
		self.filter_params = {'a_t':100, 'a_h':16, 'max_n_holes':8}
		self.vis_params = {'vis_level':-1, 'line_thickness':250}
		self.patch_params = {'use_padding':True, 'contour_fn':'four_pt'}
		self.patch_level = 0
	
	def __call__(self):
		slides = sorted(os.listdir(self.dpath_mrxs))
		slides = [
			slide for slide in slides
			if os.path.isfile(os.path.join(self.dpath_mrxs, slide))
		]

		df = initialize_df(slides, self.seg_params, self.filter_params, self.vis_params, self.patch_params)

		mask = df['process'] == 1
		process_stack = df[mask]
		total = len(process_stack)

		for i in tqdm(range(total)):
			df.to_csv(os.path.join(self.dpath_patchRoot, 'process_list_autogen.csv'), index=False)

			idx = process_stack.index[i]
			slide = process_stack.loc[idx, 'slide_id']
			#print('processing {}'.format(slide))
			
			df.loc[idx, 'process'] = 0
			slide_id, _ = os.path.splitext(slide)

			# Inialize WSI
			slide_path = os.path.join(self.dpath_mrxs, slide)
			WSI_object = WholeSlideImage(slide_path)

			current_vis_params = {}
			for key in self.vis_params.keys():
				current_vis_params.update({key: df.loc[idx, key]})

			current_filter_params = {}
			for key in self.filter_params.keys():
				current_filter_params.update({key: df.loc[idx, key]})

			current_seg_params = {}
			for key in self.seg_params.keys():
				current_seg_params.update({key: df.loc[idx, key]})

			current_patch_params = {}
			for key in self.patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})
			
			if current_vis_params['vis_level'] < 0:
				if len(WSI_object.level_dim) == 1:
					current_vis_params['vis_level'] = 0
				else:	
					wsi = WSI_object.getOpenSlide()
					best_level = wsi.get_best_level_for_downsample(64)
					current_vis_params['vis_level'] = best_level

			if current_seg_params['seg_level'] < 0:
				if len(WSI_object.level_dim) == 1:
					current_seg_params['seg_level'] = 0
				else:
					wsi = WSI_object.getOpenSlide()
					best_level = wsi.get_best_level_for_downsample(64)
					current_seg_params['seg_level'] = best_level

			keep_ids = str(current_seg_params['keep_ids'])
			if keep_ids != 'none' and len(keep_ids) > 0:
				str_ids = current_seg_params['keep_ids']
				current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
			else:
				current_seg_params['keep_ids'] = []

			exclude_ids = str(current_seg_params['exclude_ids'])
			if exclude_ids != 'none' and len(exclude_ids) > 0:
				str_ids = current_seg_params['exclude_ids']
				current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
			else:
				current_seg_params['exclude_ids'] = []

			w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
			if w * h > 1e8:
				print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
				df.loc[idx, 'status'] = 'failed_seg'
				continue

			df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
			df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

			WSI_object.segmentTissue(**current_seg_params, filter_params=current_filter_params)

			mask = WSI_object.visWSI(**current_vis_params)

			#mask_name = f"{slide_id}_{current_seg_params['seg_level']}_{current_seg_params['sthresh']}_{current_seg_params['mthresh']}_{current_seg_params['close']}_{int(current_seg_params['use_otsu'])}.jpg"
			mask_name = f"{slide_id}.jpg"
			mask_path = os.path.join(
				self.dpath_patchset_masks,
				mask_name,
			)
			mask.save(mask_path)
			"""
			"""

			current_patch_params.update({
				'patch_level': self.patch_level,
				'patch_size': self.patch_size,
				'step_size': self.step_size, 
				'save_path': self.dpath_patchset
			})
			file_path = WSI_object.process_contours(**current_patch_params)

			file_path = os.path.join(self.dpath_patchset, slide_id+'.h5')
			if os.path.isfile(file_path):
				heatmap = StitchCoords(file_path, WSI_object, downscale=64, bg_color=(0,0,0), alpha=-1, draw_grid=False)
				#stitch_path = os.path.join(self.dpath_patchset_stitch, slide_id+'.jpg')
				#heatmap.save(stitch_path)
				
				heatmap.save(os.path.join(
					self.dpath_patchset_stitch,
					'current.jpg'
				))
			
			"""
			##########################
			if os.path.isfile(file_path):
				mask = np.array(WSI_object.visWSI(**current_vis_params))
				heatmap = np.array(StitchCoords(file_path, WSI_object, downscale=64, bg_color=(0,0,0), alpha=-1, draw_grid=False))
				assem = Image.fromarray(np.concatenate([mask, heatmap], axis=1))
				dpath_assem = f"../diagnostics/{current_seg_params['sthresh']}_{current_seg_params['mthresh']}_{current_seg_params['close']}_{int(current_seg_params['use_otsu'])}"
				if not os.path.isdir(dpath_assem):
					os.makedirs(dpath_assem)
				assem_name = f"{slide_id.rstrip('.mrsx')}.jpg"
				fpath_assem = os.path.join(dpath_assem, assem_name)
				assem.save(fpath_assem)
			##########################
			"""

			df.loc[idx, 'status'] = 'processed'


		df.to_csv(os.path.join(self.dpath_patchRoot, 'process_list_autogen.csv'), index=False)

