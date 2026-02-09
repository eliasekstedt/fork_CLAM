
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords

from tqdm import tqdm
import pandas as pd

class CoordGenerator:
	def __init__(self, dpath_wsiRoot, dpath_wsiCoordRoot, dpath_wsiCoord,
		dpath_mask, dpath_stitch, fpath_segmParam,
		fpath_segmlog,
	):
		self.dpath_wsiRoot = dpath_wsiRoot
		self.dpath_patchRoot = dpath_wsiCoordRoot
		self.dpath_coord = dpath_wsiCoord
		self.dpath_mask = dpath_mask
		self.dpath_stitch = dpath_stitch
		#segm params
		self.max_holes = 800
		#patch params
		self.patch_size = 256
		self.step_size = 256
		self.patch_level = 0

		self.vis_params = {'vis_level':-1, 'line_thickness':250}

		self.fpath_segmParam = fpath_segmParam
		self.fpath_segmlog = fpath_segmlog

	def __call__(self):
		if self.fpath_segmlog.is_file():
			segmlog = pd.read_csv(self.fpath_segmlog)
		else:
			segmlog = pd.read_csv(self.fpath_segmParam)
			segmlog = segmlog[segmlog['category'] == 1]
			segmlog['handled'] = 0

		for _, row in tqdm(segmlog.copy().iterrows(), total=segmlog.shape[0]):
			assert row['category'] == 1
			slide_id = row['slide_id']
			if row['handled'] == 1:
				print(f"*** {slide_id} already handled")
				continue

			mthresh = row['mthresh']
			close = row['close']
			a_t = row['a_t']
			a_h = row['a_h']

			# Inialize WSI
			fpath_slide = self.dpath_wsiRoot / f"patient_{slide_id}.mrxs"
			WSI_object = WholeSlideImage(str(fpath_slide))
			best_downsample_lvl = WSI_object.wsi.get_best_level_for_downsample(64)
			self.vis_params['vis_level'] = best_downsample_lvl

			w, h = WSI_object.level_dim[best_downsample_lvl] 
			assert w * h < 1e8

			WSI_object.segmentTissue(
				seg_level=best_downsample_lvl,
				mthresh=mthresh,
				close=close,
				a_t=a_t,
				a_h=a_h,
				max_holes=self.max_holes,
			)

			mask = WSI_object.visWSI(**self.vis_params)
			mask_name = f"{slide_id}.jpg"
			mask.save(self.dpath_mask / mask_name)

			WSI_object.process_contours(
				save_path=str(self.dpath_coord),
				patch_level=self.patch_level,
				patch_size=self.patch_size,
				step_size=self.step_size,
			)
			
			fpath_coord = self.dpath_coord / f"{slide_id}.h5"
			if fpath_coord.is_file():
				heatmap = StitchCoords(str(fpath_coord), WSI_object, downscale=64, bg_color=(0,0,0), alpha=-1, draw_grid=False)
				heatmap.save(self.dpath_stitch / f"{slide_id}.jpg")
				segmlog.loc[segmlog['slide_id']==slide_id, 'handled'] = 1
				segmlog.to_csv(self.fpath_segmlog, index=False)



