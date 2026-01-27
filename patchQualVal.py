
from pbo_config import cfg
from pathlib import Path
import openslide
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

class Validator:
    def __init__(self, dpath_mrxs, dpath_coords, dpath_samples):
        for fpath_mrxs in tqdm(dpath_mrxs.iterdir()):
            if not fpath_mrxs.is_file():
                continue

            fpath_coords = dpath_coords / fpath_mrxs.name.replace('.mrxs', '.h5')
            fpath_sample = dpath_samples / fpath_mrxs.name.replace('.mrxs', '.png').lstrip('patient_')
            if fpath_sample.is_file():
                with open(dpath_samples / 'skip.txt', 'a') as file:
                    file.write(f"skipping {fpath_mrxs}")
                continue
            
            wsi = openslide.open_slide(fpath_mrxs)
            coords, patch_size, lvl = self.get_sample_meta(fpath_coords)
            sample = self.assemble_sample(wsi, coords, patch_size, lvl)
            Image.fromarray(sample).save(fpath_sample)
            
    def assemble_sample(self, wsi, coords, size, lvl):
        assem, row = None, None
        for pos in coords:
            patch = np.array(wsi.read_region(
                location=pos,
                level=lvl,
                size=(size, size)
            ).convert('RGB'))
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
        return assem

    def get_sample_meta(self, fpath_coords, n=25):
        coords_for_sample_patches = []
        with h5py.File(fpath_coords, "r") as f:
            avail_coords = f['coords']
            #print(avail_coords.shape)
            #raise SystemExit
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







"""
print(
    "\nmrxs: {}\ncoord: {}\nsample: {}".format(
        fpath_mrxs, fpath_coord, fpath_sample
))
"""

"""
for name in coord_names:
    assert (dpath_coords / name).is_file()
[print(i, wsin, cn) for i, (wsin, cn) in enumerate(zip(wsi_names, coord_names))]
"""
"""
fpath_coords = [

]
for fpath in fpaths_wsi:
    wsi_obj = WholeSlideImage(fpath)
    self.gen_wsi_sample(wsi_obj)
"""