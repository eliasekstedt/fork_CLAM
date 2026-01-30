
from pbo_config import cfg
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import openslide
import h5py
import numpy as np
import cv2

from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

def gshow(tensor):
    plt.imshow(tensor, cmap='grey')
    plt.show()

class Validator:
    def __init__(self, dpath_mrxs, dpath_coords, dpath_samples):
        for fpath_mrxs in tqdm(dpath_mrxs.iterdir(), total=len(list(dpath_mrxs.iterdir()))):
            if not fpath_mrxs.is_file():
                continue

            fpath_coords = dpath_coords / fpath_mrxs.name.replace('.mrxs', '.h5')
            fpath_sample = dpath_samples / fpath_mrxs.name.replace('.mrxs', '.png').lstrip('patient_')
            """
            if fpath_sample.is_file():
                with open(dpath_samples / 'skip.txt', 'a') as file:
                    file.write(f"skipping {fpath_mrxs}")
                continue
            """
            
            wsi = openslide.open_slide(fpath_mrxs)
            coords, patch_size, lvl = self.get_sample_meta(fpath_coords)
            sample = self.assemble_sample(wsi, coords, patch_size, lvl)
            #Image.fromarray(sample).save(fpath_sample)
            
    def assemble_sample(self, wsi, coords, size, lvl):
        def process_patch(patch):
            patch = cv2.cvtColor(patch[:, :, ::-1], cv2.COLOR_BGR2GRAY)
            patch = cv2.GaussianBlur(patch, (15, 15), sigmaX=0)
            binary = np.array(patch < 245).astype(np.int16) * 255

            kernel = np.ones((3, 3), np.uint8)
            morphed_e = cv2.erode(binary, kernel, iterations=3)
            morphed_d = cv2.dilate(morphed_e, kernel, iterations=1)
            print(np.mean(np.where(morphed_d > 0, 1, 0)))
            gshow(np.concatenate([
                np.concatenate([patch, binary], axis=1),
                np.concatenate([morphed_d, morphed_e], axis=1),
            ], axis=0))
            """
            """
            

        def read_patch(wsi, pos, lvl, size):
            patch = np.array(wsi.read_region(
                location=pos,
                level=lvl,
                size=(size, size)
            ).convert('RGB'))
            return patch
        
        assem, row = None, None
        for pos in coords:
            patch = read_patch(wsi, pos, lvl, size)

            if True:
                process_patch(patch)

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
        plt.imshow(assem)
        plt.show()
        raise SystemExit
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

dpath_mrxsRoot = Path(cfg.dpath_mrxsRoot)
dpath_coords = Path(cfg.dpath_patchset)

Validator(
    dpath_mrxs=dpath_mrxsRoot,
    dpath_coords=dpath_coords,
    dpath_samples=cfg.dpath_patch2encode_samples,
)

