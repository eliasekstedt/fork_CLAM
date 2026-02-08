
# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
#from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
from tqdm import tqdm
#import time
#import pandas as pd
from PIL import Image

class PatchCoordMapper:
	def __init__(self, dpath_mrxs, dpath_patchRoot, dpath_patchset, dpath_patchset_masks,
		dpath_patchset_stitch, mthresh, close, a_t, a_h, max_holes, live,
	):
		self.live = live
		self.dpath_mrxs = dpath_mrxs
		self.dpath_patchRoot = dpath_patchRoot
		self.dpath_patchset = dpath_patchset
		self.dpath_patchset_masks = dpath_patchset_masks
		self.dpath_patchset_stitch = dpath_patchset_stitch
		#seg params
		self.mthresh = mthresh
		self.close = close
		# filter params
		self.a_t = a_t
		self.a_h = a_h
		self.max_holes = max_holes
		#patch params
		self.patch_size = 256
		self.step_size = 256
		self.patch_level = 0

		self.vis_params = {'vis_level':-1, 'line_thickness':250}

	def __call__(self):
		slides = sorted(os.listdir(self.dpath_mrxs))
		slide_ids = [
			slide for slide in slides
			if os.path.isfile(os.path.join(self.dpath_mrxs, slide))
			and not slide == '.DS_Store'
		]
		
		for slide_id in tqdm(slide_ids):
			# Inialize WSI
			slide_path = os.path.join(self.dpath_mrxs, slide_id)
			WSI_object = WholeSlideImage(slide_path)
			best_downsample_lvl = WSI_object.wsi.get_best_level_for_downsample(64)
			self.vis_params['vis_level'] = best_downsample_lvl

			w, h = WSI_object.level_dim[best_downsample_lvl] 
			assert w * h < 1e8

			WSI_object.segmentTissue(
				seg_level=best_downsample_lvl,
				mthresh=self.mthresh,
				close=self.close,
				a_t=self.a_t,
				a_h=self.a_h,
				max_holes=self.max_holes,
			)

			if self.live:
				mask = WSI_object.visWSI(**self.vis_params)
				mask_name = f"{slide_id.rstrip('.mrsx')}.jpg"
				mask_path = os.path.join(
					self.dpath_patchset_masks,
					mask_name,
				)
				mask.save(mask_path)

			WSI_object.process_contours(
				save_path=self.dpath_patchset,
				patch_level=self.patch_level,
				patch_size=self.patch_size,
				step_size=self.step_size,
			)
			file_path = os.path.join(self.dpath_patchset, slide_id.rstrip('.mrxs')+'.h5')

			if os.path.isfile(file_path):
				if self.live:
					heatmap = StitchCoords(file_path, WSI_object, downscale=64, bg_color=(0,0,0), alpha=-1, draw_grid=False)
					stitch_path = os.path.join(self.dpath_patchset_stitch, slide_id.rstrip('.mrxs')+'.jpg')
					heatmap.save(stitch_path)
				else:
					mask = np.array(WSI_object.visWSI(**self.vis_params))
					heatmap = np.array(StitchCoords(file_path, WSI_object, downscale=64, bg_color=(0,0,0), alpha=-1, draw_grid=False))
					assem = Image.fromarray(np.concatenate([mask, heatmap], axis=1))
					dpath_assem = f"../data/biops_diagnostics/the_return_0/"
					if not os.path.isdir(dpath_assem):
						os.makedirs(dpath_assem)
					assem_name = f"{slide_id.rstrip('.mrsx')}.jpg"
					fpath_assem = os.path.join(dpath_assem, assem_name)
					assem.save(fpath_assem)
					raise SystemExit
