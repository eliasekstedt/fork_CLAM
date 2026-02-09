
from wsi_core.WholeSlideImage import WholeSlideImage
import pandas as pd
import numpy as np
import cv2
import openslide
import h5py
from pathlib import Path
from tqdm import tqdm

class PatchFilter:
    def __init__(self):
        pass

    def on_blot(self, patch):
        r, g, b = np.array_split(patch, patch.shape[-1], axis=2)
        blood_for_the_blood_god = np.where(r > g + b, 1, 0)

    def on_blur(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
        
    def on_bg(self, patch):
        patch = cv2.cvtColor(patch[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        patch = cv2.GaussianBlur(patch, (15, 15), sigmaX=0)
        patch = np.array(patch < 240).astype(np.int16) * 255
        return np.mean(np.where(patch > 0, 1, 0))
    
    def on_dist(self, patch):
        ref = np.ones_like(patch) * 255
        return np.mean(ref - patch)

    def apply(self, patch):
        patch = np.array(patch)
        return self.on_bg(patch), self.on_blur(patch), self.on_dist(patch)
    

class QualityAssigner:
    def __init__(self, wsi, fpath_wsiCoords, fpath_qualityLog, fltr):
        slide_id = fpath_wsiCoords.name.rstrip('.csv')
        with h5py.File(fpath_wsiCoords, "r") as file:
            coords = file['coords'][:]
            patch_level = file['coords'].attrs['patch_level']
            patch_size = file['coords'].attrs['patch_size']

        rows = []
        for idx in range(coords.shape[0]):
            pos = coords[idx]
            img = wsi.read_region(pos, patch_level, (patch_size, patch_size)).convert('RGB')
            on_bg, on_blur, on_dist = fltr.apply(img)
            rows.append({
                'pos_x':pos[0],
                'pos_y':pos[1],
                'on_bg':on_bg,
                'on_blur':on_blur,
                'on_dist':on_dist,
            })
        df = pd.DataFrame(rows)
        df['slide_id'] = slide_id
        df['patch_lvl'] = patch_level
        df['patch_size'] = patch_size
        df = df[['slide_id', 'pos_x', 'pos_y', 'on_bg', 'on_blur', 'on_dist', 'patch_lvl', 'patch_size']]
        df.to_csv(fpath_qualityLog, index=False)
    
class QALooper:
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
            QualityAssigner(
                wsi=wsi,
                fpath_wsiCoords=fpath_wsiCoords,
                fpath_qualityLog=fpath_qualityLog,
                fltr=PatchFilter(),
            )

