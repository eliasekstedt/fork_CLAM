
import pandas as pd
import numpy as np
import cv2
import openslide
import h5py
from pathlib import Path
from tqdm import tqdm


class QualityMeter:
    def __init__(self):
        pass

    def on_bg(self, patch):
        blurred = cv2.GaussianBlur(patch, (15, 15), 0)

        gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)

        # Faster than full std
        stdmask = blurred.std(axis=2)

        min_std = stdmask.min()
        max_gray = gray.max()

        condition_0 = stdmask < (min_std + 3) * 1.1
        condition_1 = gray > max_gray * 0.96

        mask = condition_0 & condition_1
        return mask.mean(), mask


    def on_bg_slow(self, patch):
        bpatch = cv2.GaussianBlur(patch, (15, 15), sigmaX=0)
        gpatch = cv2.cvtColor(bpatch[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        stdmask = np.std(bpatch, axis=2)
        condition_0 = np.where(stdmask < (np.min(stdmask)+3) * 1.1, 255, 0)
        condition_1 = np.where(gpatch > np.max(gpatch) * 0.96, 255, 0)
        mask = np.logical_and(condition_0, condition_1)
        return np.mean(mask), mask
    
    def on_blur(self, patch, mask):
        mask = ~mask
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = lap[mask > 0]
        
        if lap.size == 0:
            return 0 
            
        return lap.var()
    
    def apply(self, patch):
        patch = np.array(patch)
        bg_score, mask = self.on_bg(patch)
        blur_score = self.on_blur(patch=patch, mask=mask)
        return blur_score, bg_score


class QMAssigner:
    def __init__(self, wsi, fpath_wsiCoords, fpath_qualityLog, quality_meter):
        slide_id = fpath_wsiCoords.name.rstrip('.h5')
        with h5py.File(fpath_wsiCoords, "r") as file:
            coords = file['coords'][:]
            patch_level = file['coords'].attrs['patch_level']
            patch_size = file['coords'].attrs['patch_size']

        rows = []
        for idx in range(coords.shape[0]):
            pos = coords[idx]
            patch = np.array(wsi.read_region(pos, patch_level, (patch_size, patch_size)).convert('RGB'))

            blur, bg = quality_meter.apply(patch)
            patch = patch.astype(np.float32)

            rows.append({
                'pos_x':pos[0],
                'pos_y':pos[1],
                'blur':blur,
                'bg':bg,
            })

        df = pd.DataFrame(rows)
        df['slide_id'] = slide_id
        df['patch_lvl'] = patch_level
        df['patch_size'] = patch_size
        df = df[['slide_id', 'pos_x', 'pos_y', 'blur', 'bg', 'patch_lvl', 'patch_size']]
        df.to_csv(fpath_qualityLog, index=False)
    
class QMAWrapper:
    def __init__(self, dpath_wsiRoot, dpath_qualityLog, dpath_wsiCoords, fpath_segmlog):
        segmlog = pd.read_csv(fpath_segmlog)
        slide_ids = segmlog['slide_id'].to_list()

        for slide_id in tqdm(slide_ids, total=len(slide_ids)):
            fpath_qualityLog = dpath_qualityLog / f'{slide_id}.csv'
            if fpath_qualityLog.is_file():
                print(f"*** {slide_id} already handled")
                continue

            fpath_slide = Path(dpath_wsiRoot / f'patient_{slide_id}.mrxs')
            fpath_wsiCoords = dpath_wsiCoords / f'{slide_id}.h5'
            wsi = openslide.open_slide(fpath_slide)

            QMAssigner(
                wsi=wsi,
                fpath_wsiCoords=fpath_wsiCoords,
                fpath_qualityLog=fpath_qualityLog,
                quality_meter=QualityMeter(),
            )

