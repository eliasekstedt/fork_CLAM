
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

        """
        diff = np.abs(condition_0.sum() - condition_1.sum())
        if diff > self.maxdiff:
            print(self.maxdiff, diff)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].imshow(patch)
            ax[0, 0].set_xticks([])
            ax[0, 0].set_yticks([])
            ax[0, 1].imshow(mask, cmap='grey')
            ax[0, 1].set_xticks([])
            ax[0, 1].set_yticks([])
            ax[1, 0].imshow(condition_0, cmap='grey')
            ax[1, 0].set_xticks([])
            ax[1, 0].set_yticks([])
            ax[1, 1].imshow(condition_1, cmap='grey')
            ax[1, 1].set_xticks([])
            ax[1, 1].set_yticks([])
            plt.tight_layout()
            plt.show()
            self.maxdiff = diff
        """

        return mask.mean(), mask


    def on_bg_slow(self, patch):
        bpatch = cv2.GaussianBlur(patch, (15, 15), sigmaX=0)
        gpatch = cv2.cvtColor(bpatch[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        stdmask = np.std(bpatch, axis=2)
        condition_0 = np.where(stdmask < (np.min(stdmask)+3) * 1.1, 255, 0)
        condition_1 = np.where(gpatch > np.max(gpatch) * 0.96, 255, 0)
        mask = np.logical_and(condition_0, condition_1)
        return np.mean(mask), mask

    """
    def on_blur(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    """
    
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


class MetaAssigner:
    def __init__(self, wsi, fpath_wsiCoords, fpath_qualityLog, fltr):
        slide_id = fpath_wsiCoords.name.rstrip('.h5')
        with h5py.File(fpath_wsiCoords, "r") as file:
            coords = file['coords'][:]
            patch_level = file['coords'].attrs['patch_level']
            patch_size = file['coords'].attrs['patch_size']

        fltr = PatchFilter()

        rows = []
        for idx in range(coords.shape[0]):
            pos = coords[idx]
            patch = np.array(wsi.read_region(pos, patch_level, (patch_size, patch_size)).convert('RGB'))

            blur, bg = fltr.apply(patch)
            patch = patch.astype(np.float32)

            rows.append({
                'pos_x':pos[0],
                'pos_y':pos[1],
                'blur':blur,
                'bg':bg,
                'pix_sum':np.sum(patch, axis=(0, 1)),
                'pix_sq_sum':np.sum(patch**2, axis=(0, 1)),
                'n_pixls':patch.shape[0] * patch.shape[1],
            })

        df = pd.DataFrame(rows)
        df['slide_id'] = slide_id
        df['patch_lvl'] = patch_level
        df['patch_size'] = patch_size
        df = df[['slide_id', 'pos_x', 'pos_y', 'blur', 'bg', 'patch_lvl', 'patch_size']]
        df.to_csv(fpath_qualityLog, index=False)
    
class MetaLooper:
    def __init__(self, dpath_wsiRoot, dpath_qualityLog, dpath_wsiCoords, fpath_segmlog):
        segmlog = pd.read_csv(fpath_segmlog)
        slide_ids = segmlog['slide_id'].to_list()
        #i = 0
        for slide_id in tqdm(slide_ids, total=len(slide_ids)):
            fpath_qualityLog = dpath_qualityLog / f'{slide_id}.csv'
            if fpath_qualityLog.is_file():
                print(f"*** {slide_id} already handled")
                continue

            #i += 1
            #if i > 50:
            #    break

            fpath_slide = Path(dpath_wsiRoot / f'patient_{slide_id}.mrxs')
            fpath_wsiCoords = dpath_wsiCoords / f'{slide_id}.h5'
            wsi = openslide.open_slide(fpath_slide)

            MetaAssigner(
                wsi=wsi,
                fpath_wsiCoords=fpath_wsiCoords,
                fpath_qualityLog=fpath_qualityLog,
                fltr=PatchFilter(),
            )

"""
def on_otsu(self, patch):
    def get_chan_std(patch, mask):
        tissue_or_zero = patch * mask[:, :, None]
        tissue_chan_std = np.std(np.sum(tissue_or_zero, axis=(0, 1)) / (np.sum(mask)*255))
        return tissue_chan_std
    
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    patch_s = cv2.medianBlur(patch_hsv[:, :, 1], 3)
    _, mask = cv2.threshold(patch_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask // 255

    tissue_chan_std = get_chan_std(patch, mask)
    not_tissue_chan_std = get_chan_std(patch, np.ones_like(mask) - mask)

    return np.mean(mask), tissue_chan_std, not_tissue_chan_std
"""