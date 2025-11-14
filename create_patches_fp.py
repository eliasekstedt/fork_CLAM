# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import pandas as pd
from tqdm import tqdm


class PatchsetGenerator:
	"""
	source: dpath_mrxsDir
	save_dir:patchsetDir
	patch_save_dir: ...
	mask_save_dir: ...
	stitch_save_dir: ...
	"""
	def __init__(self, dpath_mrxsDir, dpath_patchsetDir):
		self.dpath_mrxsDir = dpath_mrxsDir
		self.dpath_patchsetDir = dpath_patchsetDir
		self.patch_save_dir = os.path.join(dpath_patchsetDir, 'patches')
		self.mask_save_dir = os.path.join(dpath_patchsetDir, 'masks')
		self.stitch_save_dir = os.path.join(dpath_patchsetDir, 'stitches')
		self.patch_size = 256
		self.step_size = 256
		self.seg_params = {
			'seg_level': -1,
			'sthresh': 8,
			'mthresh': 7,
			'close': 4,
			'use_otsu': False,
			'keep_ids': 'none',
			'exclude_ids': 'none'
		}
		self.filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
		self.vis_params = {'vis_level': -1, 'line_thickness': 500},
		self.patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
		self.patch_level = 0,
		self.use_default_params = False,
		self.seg = False,
		self.save_mask = True,
		self.stitch= False,
		self.patch = False,
		self.auto_skip=True,
		self.process_list = None
	
	def __call__(self):
		slides = sorted(os.listdir(self.dpath_mrxsDir))
		slides = [
			slide for slide in slides
			if os.path.isfile(os.path.join(self.dpath_mrxsDir, slide))
		]
		if self.process_list is None:
			df = initialize_df(slides, self.seg_params, self.filter_params, self.vis_params, self.patch_params)
		
		else:
			df = pd.read_csv(self.process_list)
			df = initialize_df(df, self.seg_params, self.filter_params, self.vis_params, self.patch_params)

		mask = df['process'] == 1
		process_stack = df[mask]

		total = len(process_stack)

		legacy_support = 'a' in df.keys()
		if legacy_support:
			print('detected legacy segmentation csv file, legacy support enabled')
			df = df.assign(**{'a_t': np.full((len(df)), int(self.filter_params['a_t']), dtype=np.uint32),
			'a_h': np.full((len(df)), int(self.filter_params['a_h']), dtype=np.uint32),
			'max_n_holes': np.full((len(df)), int(self.filter_params['max_n_holes']), dtype=np.uint32),
			'line_thickness': np.full((len(df)), int(self.vis_params['line_thickness']), dtype=np.uint32),
			'contour_fn': np.full((len(df)), self.patch_params['contour_fn'])})

		seg_times = 0.
		patch_times = 0.
		stitch_times = 0.

		for i in tqdm(range(total)):
			df.to_csv(os.path.join(self.dpath_patchsetDir, 'process_list_autogen.csv'), index=False)
			idx = process_stack.index[i]
			slide = process_stack.loc[idx, 'slide_id']
			print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
			print('processing {}'.format(slide))
			
			df.loc[idx, 'process'] = 0
			slide_id, _ = os.path.splitext(slide)

			if self.auto_skip and os.path.isfile(os.path.join(self.patch_save_dir, slide_id + '.h5')):
				print('{} already exist in destination location, skipped'.format(slide_id))
				df.loc[idx, 'status'] = 'already_exist'
				continue

			# Inialize WSI
			full_path = os.path.join(self.dpath_mrxsDir, slide)
			WSI_object = WholeSlideImage(full_path)

			if self.use_default_params:
				current_vis_params = self.vis_params.copy()
				current_filter_params = self.filter_params.copy()
				current_seg_params = self.seg_params.copy()
				current_patch_params = self.patch_params.copy()
				
			else:
				current_vis_params = {}
				current_filter_params = {}
				current_seg_params = {}
				current_patch_params = {}

				for key in self.vis_params.keys():
					if legacy_support and key == 'vis_level':
						df.loc[idx, key] = -1
					current_vis_params.update({key: df.loc[idx, key]})

				for key in self.filter_params.keys():
					if legacy_support and key == 'a_t':
						old_area = df.loc[idx, 'a']
						seg_level = df.loc[idx, 'seg_level']
						scale = WSI_object.level_downsamples[seg_level]
						adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
						current_filter_params.update({key: adjusted_area})
						df.loc[idx, key] = adjusted_area
					current_filter_params.update({key: df.loc[idx, key]})

				for key in self.seg_params.keys():
					if legacy_support and key == 'seg_level':
						df.loc[idx, key] = -1
					current_seg_params.update({key: df.loc[idx, key]})

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


			seg_time_elapsed = -1
			if self.seg:
				WSI_object, seg_time_elapsed = self.segment(WSI_object, current_seg_params, current_filter_params) 

			if self.save_mask:
				mask = WSI_object.visWSI(**current_vis_params)
				mask_path = os.path.join(self.mask_save_dir, slide_id+'.jpg')
				mask.save(mask_path)

			patch_time_elapsed = -1 # Default time
			if self.patch:
				current_patch_params.update({
					'patch_level': self.patch_level,
					'patch_size': self.patch_size,
					'step_size': self.step_size, 
					'save_path': self.patch_save_dir
				})
				file_path, patch_time_elapsed = self.patching(WSI_object = WSI_object,  **current_patch_params,)
			
			stitch_time_elapsed = -1
			if self.stitch:
				file_path = os.path.join(self.patch_save_dir, slide_id+'.h5')
				if os.path.isfile(file_path):
					heatmap, stitch_time_elapsed = self.stitching(file_path, WSI_object, downscale=64)
					stitch_path = os.path.join(self.stitch_save_dir, slide_id+'.jpg')
					heatmap.save(stitch_path)

			print("segmentation took {} seconds".format(seg_time_elapsed))
			print("patching took {} seconds".format(patch_time_elapsed))
			print("stitching took {} seconds".format(stitch_time_elapsed))
			df.loc[idx, 'status'] = 'processed'

			seg_times += seg_time_elapsed
			patch_times += patch_time_elapsed
			stitch_times += stitch_time_elapsed

		seg_times /= total
		patch_times /= total
		stitch_times /= total

		df.to_csv(os.path.join(self.dpath_patchsetDir, 'process_list_autogen.csv'), index=False)
		print("average segmentation time in s per slide: {}".format(seg_times))
		print("average patching time in s per slide: {}".format(patch_times))
		print("average stiching time in s per slide: {}".format(stitch_times))
			
		return seg_times, patch_times


	def stitching(file_path, wsi_object, downscale = 64):
		start = time.time()
		heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
		total_time = time.time() - start
		return heatmap, total_time

	def segment(self, WSI_object, seg_params = None, filter_params = None, mask_file = None):
		### Start Seg Timer
		start_time = time.time()
		# Use segmentation file
		if mask_file is not None:
			WSI_object.initSegmentation(mask_file)
		# Segment	
		else:
			WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
		### Stop Seg Timers
		seg_time_elapsed = time.time() - start_time   
		return WSI_object, seg_time_elapsed

	def patching(self, WSI_object, **kwargs):
		### Start Patch Timer
		start_time = time.time()
		# Patch
		file_path = WSI_object.process_contours(**kwargs)
		### Stop Patch Timer
		patch_time_elapsed = time.time() - start_time
		return file_path, patch_time_elapsed
