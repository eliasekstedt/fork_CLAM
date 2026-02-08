
from pbo_config import *
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import openslide
import h5py
import numpy as np
import cv2

import matplotlib.pyplot as plt

def gshow(tensor):
    plt.imshow(tensor, cmap='grey')
    plt.show()

class PatchFilter:
    def __init__(self):
        pass

    def on_blot(self, patch):
        r, g, b = np.array_split(patch, patch.shape[-1], axis=2)
        blood_for_the_blood_god = np.where(r > g + b, 1, 0)

    def on_blur(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < 45
        
    def on_bg(self, patch):
        patch = cv2.cvtColor(patch[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        patch = cv2.GaussianBlur(patch, (15, 15), sigmaX=0)
        patch = np.array(patch < 240).astype(np.int16) * 255
        return np.mean(np.where(patch > 0, 1, 0)) < 0.5
    
    def apply(self, patch):
        patch = np.array(patch)
        return self.on_bg(patch) or self.on_blur(patch)

"""
    def create_bins(self):
        bin_names = [f"bin_{d}" for d in self.bins]
        for name in bin_names:
            dpath_bin = self.filtrRoot / name
            dpath_bin.mkdir(exist_ok=True)

    def categorize(self, patch):
        reject = False
        if self.on_bg(patch):
            reject = True
        if not reject:
            score = self.on_blur(patch)
            dpath_bin = self.filtrRoot / f"bin_{int(score - score % 10)}"
            fpath_bin = dpath_bin / f"p{len(list(dpath_bin.iterdir()))}_{int(score)}.png"
            Image.fromarray(patch).save(fpath_bin)
"""



class Validator:
    def __init__(self, dpath_mrxs, dpath_coords, dpath_samples, filter):
        self.reject_count, self.count = 0, 0
        content = [item for item in dpath_mrxs.iterdir() if item.is_file()][:]
        for fpath_mrxs in tqdm(content, total=len(list(content))):

            fpath_coords = dpath_coords / fpath_mrxs.name.replace('.mrxs', '.h5')
            fpath_sample = dpath_samples / fpath_mrxs.name.replace('.mrxs', '.png').lstrip('patient_')
            
            wsi = openslide.open_slide(fpath_mrxs)
            coords, patch_size, lvl = self.get_sample_meta(fpath_coords)
            sample = self.assemble_sample(wsi, coords, patch_size, lvl, filter)
            Image.fromarray(sample).save(fpath_sample)

            
    def assemble_sample(self, wsi, coords, size, lvl, filter):
        def read_patch(wsi, pos, lvl, size):
            patch = np.array(wsi.read_region(
                location=pos,
                level=lvl,
                size=(size, size)
            ).convert('RGB'))
            return patch
        
        n = 40
        assem, row = None, None
        for pos in coords:
            patch = read_patch(wsi, pos, lvl, size)

            reject = False
            if filter.on_blur(patch):
                #patch[n:2*n, 0:n, 0] = np.zeros_like(patch[0:n, 0:n, 0])
                patch[n:2*n, 0:n, :] = np.ones_like(patch[0:n, 0:n, :]) * 240
                patch[n:2*n, n//2:n, :] = np.zeros_like(patch[0:n, n//2:n, :])
                reject = True

            if filter.on_bg(patch):
                patch[0:n, 0:n, 0] = np.zeros_like(patch[0:n, 0:n, 0])
                reject = True

            if row is None:
                row = patch
            else:
                row = np.concatenate([row, patch], axis=1)
                if row.shape[1] >= 5 * patch.shape[1]:
                    if assem is None:
                        assem = row
                    else:
                        assem = np.concatenate([assem, row], axis=0)
                    row = None
            if reject:
                self.reject_count += 1
            self.count += 1
        #plt.imshow(assem)
        #plt.show()
        #raise SystemExit
        
        return assem

    def get_sample_meta(self, fpath_coords, n=25):
        coords_for_sample_patches = []
        with h5py.File(fpath_coords, "r") as f:
            avail_coords = f['coords']
            [coords_for_sample_patches.append(avail_coords[i])
             for i in np.random.choice(range(avail_coords.shape[0]), n)]
            patch_size = f['coords'].attrs['patch_size']
            lvl = f['coords'].attrs['patch_level']
        return coords_for_sample_patches, patch_size, lvl

"""
dpath_mrxsRoot = Path(cfg.dpath_mrxsRoot)
dpath_coords = Path(cfg.dpath_patchset)

filtr = PatchFilter()
validator = Validator(
    dpath_mrxs=dpath_mrxsRoot,
    dpath_coords=dpath_coords,
    dpath_samples=cfg.dpath_patch2encode_samples,
    filter=filtr,
)

reject_rate = validator.reject_count / validator.count
print(validator.reject_count, validator.count, reject_rate)

"""
"""
notable slides:
* 046_*
* 206_AB
"""