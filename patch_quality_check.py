
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
* given filtering parameters, what patches are rejected vs pass?
* among wsi, are there any outliers in terms of the values in filtering criteria or total nr patches?
"""

class QDock:
    def __init__(self, dpath_wsiRoot, dpath_wsiCoords, dpath_qualityLog,
        dpath_diagnostics, fltr_params,
    ):
        #fpaths_qlog = dpath_qualityLog.iterdir()
        self.check_extraction_geometry(
            dpath_wsiRoot=dpath_wsiRoot,
            dpath_wsiCoords=dpath_wsiCoords,
            fpaths_qlog=list(dpath_qualityLog.iterdir()),
            dpath_diagnostics=dpath_diagnostics,
        )
        per_slide_info = self.get_per_slide_info(
            fpaths_qlog=list(dpath_qualityLog.iterdir()),
            dpath_diagnostics=dpath_diagnostics,
            ll_bg=fltr_params['ll_bg'],
            ll_blur=fltr_params['ll_blur'],
            ll_dist=fltr_params['ll_dist'],
            ul_dist=fltr_params['ul_dist'],
        )

        self.vis_keep_v_reject(
            dpath_wsiRoot=dpath_wsiRoot,
            dpath_wsiCoords=dpath_wsiCoords,
            fpaths_qlog=list(dpath_qualityLog.iterdir()),
            dpath_diagnostics=dpath_diagnostics,
            ll_bg=fltr_params['ll_bg'],
            ll_blur=fltr_params['ll_blur'],
            ll_dist=fltr_params['ll_dist'],
            ul_dist=fltr_params['ul_dist'],
        )

    def vis_keep_v_reject(self, dpath_wsiRoot, dpath_wsiCoords,
        fpaths_qlog, dpath_diagnostics, ll_bg, ll_blur, ll_dist, ul_dist,
    ):
        def build_assem(wsi, df, patch_lvl, patch_size, row_units=5):
            assem, assem_row, blank = None, None, None
            for _, row in df.iterrows():
                pos = row['pos_x'], row['pos_y']
                patch = self.read_patch(wsi, pos, patch_lvl, patch_size)

                if assem_row is None:
                    assem_row = patch
                else:
                    assem_row = np.concatenate([assem_row, patch], axis=1)
                    if assem_row.shape[1] >= patch.shape[1] * row_units:
                        if assem is None:
                            assem = assem_row
                        else:
                            assem = np.concatenate([assem, assem_row], axis=0)
                        assem_row = None

                        if assem.shape[0] >= assem.shape[1]:
                            break
            return assem
            
        fpath_qlogs = np.random.choice(fpaths_qlog, 9)
        for fpath_qlog in fpath_qlogs:
            slide_id = fpath_qlog.name.rstrip('.csv')

            wsi = openslide.open_slide(dpath_wsiRoot / f"patient_{slide_id}.mrxs")
            patch_lvl, patch_size, _ = self.get_wsi_meta(dpath_wsiCoords / f"{slide_id}.h5")

            qlog = pd.read_csv(fpath_qlog)
            rejects = qlog[
                (qlog['on_bg'] < ll_bg) |
                (qlog['on_blur'] < ll_blur) |
                (qlog['on_dist'] < ll_dist) |
                (qlog['on_dist'] > ul_dist)
            ].sample(frac=1)
            reject_assem = build_assem(wsi, rejects, patch_lvl, patch_size)

            keep = qlog[~qlog.index.isin(rejects.index)].sample(frac=1)
            keep_assem = build_assem(wsi, keep, patch_lvl, patch_size)
            div = np.zeros_like(keep_assem[:, :int(keep_assem.shape[1]*0.01), :])
            assem = np.concatenate([keep_assem, div, reject_assem], axis=1)
            Image.fromarray(assem).save(dpath_diagnostics / f'qdock_keep_v_reject_{slide_id}.png')

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
            
        fpath_qlogs = np.random.choice(fpaths_qlog, 3)
        for fpath_qlog in fpath_qlogs:
            slide_id = fpath_qlog.name.rstrip('.csv')
            qlog = pd.read_csv(fpath_qlog).sort_values(by=['pos_x', 'pos_y'])
            
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
        
    def get_per_slide_info(self, fpaths_qlog, dpath_diagnostics, ll_bg, ll_blur, ll_dist, ul_dist):
        def gen_scatter_matrix(df, dpath_diagnostics):
            plotcols = ['on_bg', 'on_blur', 'on_dist', 'n_tot']
            sns.pairplot(df[plotcols], plot_kws={'s': 10, 'alpha':0.5})
            plt.tight_layout()
            plt.savefig(dpath_diagnostics / 'qdock_scatter_matrix.png')

        per_slide_info = []
        for fpath in fpaths_qlog:
            qlog = pd.read_csv(fpath)
            ntot = qlog.shape[0]
            per_slide_info.append({
                'slide_id':fpath.name.rstrip('.csv'),
                'on_bg':qlog['on_bg'].mean().item(),
                'on_blur':qlog['on_blur'].mean().item(),
                'on_dist':qlog['on_dist'].mean().item(),
                'bg_reject_rate':np.sum(qlog['on_bg'] < ll_bg) / ntot,
                'blur_reject_rate':np.sum(qlog['on_blur'] < ll_blur) / ntot,
                'ldist_reject_rate':np.sum(qlog['on_dist'] < ll_dist) / ntot,
                'udist_reject_rate':np.sum(qlog['on_dist'] > ul_dist) / ntot,
                'n_tot':ntot,
            })
            
        per_slide_info = pd.DataFrame(per_slide_info)
        gen_scatter_matrix(per_slide_info, dpath_diagnostics)
        per_slide_info.to_csv(dpath_diagnostics / 'per_slide_info.csv', index=False)
        return per_slide_info
        