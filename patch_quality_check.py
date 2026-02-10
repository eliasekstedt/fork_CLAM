
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import openslide
import h5py
from PIL import Image
"""
questions to answer:
* are patches overlapping or skipping area?
* given filtering params, for each wsi, what patches are rejected or approved?
* given filtering parameters, what patches are rejected vs pass?
* among wsi, are there any outliers in terms of the values in filtering criteria or total nr patches?

"""

class QDock:
    def __init__(self, dpath_wsiRoot, dpath_wsiCoords, dpath_qualityLog,
        dpath_diagnostics,
    ):
        fpaths_qlog = dpath_qualityLog.iterdir()
        #self.gen_scatter_matrix(fpaths_qlog, dpath_diagnostics)
        self.check_extraction_geometry(dpath_wsiRoot, dpath_wsiCoords, fpaths_qlog, dpath_diagnostics)

    def read_patch(self, wsi, pos, patch_level, patch_size):
        return np.array(wsi.read_region(pos, patch_level, (patch_size, patch_size)).convert('RGB'))

    def get_wsi_meta(self, fpath_wsiCoords):
        with h5py.File(fpath_wsiCoords, "r") as file:
            coords = file['coords'][:]
            patch_lvl = file['coords'].attrs['patch_level']
            patch_size = file['coords'].attrs['patch_size']
        return patch_lvl, patch_size, coords
        
    def check_extraction_geometry(self, dpath_wsiRoot, dpath_wsiCoords, fpaths_qlog, dpath_diagnostics):
        def build_assem(which_x, which_y, qlog, wsi, patch_level, patch_size):
            blank = np.zeros_like(self.read_patch(
                pos=(which_x[len(which_x) // 2], which_y[len(which_y) // 2]),
                wsi=wsi,
                patch_level=patch_level,
                patch_size=patch_size,
            ))

            assem, assem_row = None, None
            for y in which_y:
                for x in which_x:
                    if qlog[(qlog['pos_x']==x) & (qlog['pos_y']==y)].empty:
                        patch = blank
                    else:
                        patch = self.read_patch(
                            pos=(x, y),
                            wsi=wsi,
                            patch_level=patch_level,
                            patch_size=patch_size,
                        )
                    if assem_row is None:
                        assem_row = patch
                    else:
                        assem_row = np.concatenate([assem_row, patch], axis=1)
                        if assem_row.shape[1] >= patch.shape[1] * len(which_x):
                            if assem is None:
                                assem = assem_row
                            else:
                                assem = np.concat([assem, assem_row], axis=0)
                            assem_row = None
            return assem
            
        fpath_qlogs = np.random.choice(list(fpaths_qlog), 3)
        for fpath_qlog in fpath_qlogs:
            slide_id = fpath_qlog.name.rstrip('.csv')
            qlog = pd.read_csv(fpath_qlog).sort_values(by=['pos_x', 'pos_y'])
            
            #xcentr_subset = qlog[qlog['pos_x'] == qlog['pos_x'].median()]
            xcentr_subset = qlog[qlog['pos_x'] == qlog['pos_x'].iloc[qlog.shape[0]//2]]
            if xcentr_subset.empty:
                print(f'no_geocheck for {slide_id}')
                continue

            x_centr = xcentr_subset['pos_x'].unique().item()
            y_centr = int(xcentr_subset['pos_y'].iloc[xcentr_subset.shape[0]//2])

            patch_lvl, patch_size, _ = self.get_wsi_meta(dpath_wsiCoords / f"{slide_id}.h5")

            which_x = [(x_centr - patch_size).item(), x_centr, (x_centr + patch_size).item()]
            which_y = [(y_centr - patch_size).item(), y_centr, (y_centr + patch_size).item()]

            wsi = openslide.open_slide(dpath_wsiRoot / f"patient_{slide_id}.mrxs")
            assem = build_assem(which_x, which_y, qlog, wsi, patch_lvl, patch_size)

            Image.fromarray(assem.astype(np.uint8)).save(dpath_diagnostics / f'qdock_geoCheck_{slide_id}.png')
        

    def gen_scatter_matrix(self, fpaths_qlog, dpath_diagnostics):
        quality_stats = []
        for fpath in fpaths_qlog:
            qlog = pd.read_csv(fpath)
            quality_stats.append({
                'slide_id':fpath.name.rstrip('.csv'),
                'on_bg':qlog['on_bg'].mean().item(),
                'on_blur':qlog['on_blur'].mean().item(),
                'on_dist':qlog['on_dist'].mean().item(),
                'n_tot':qlog.shape[0],
            })
            
        quality_stats = pd.DataFrame(quality_stats)
        sns.pairplot(quality_stats)
        plt.savefig(dpath_diagnostics / 'qdock_scatter_matrix.png')


ll_bg = 0.5
ll_blur = 0.2
ll_dist = 20
ul_dist = 230

from config import *
QDock(
    dpath_wsiRoot=cfg.dpath_wsiRoot,
    dpath_wsiCoords=cfg.dpath_wsiCoords,
    dpath_qualityLog=cfg.dpath_qualityLog,
    dpath_diagnostics=cfg.dpath_diagnostics,
)
