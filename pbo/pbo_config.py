
import os

def create_dirs(*dirpaths):
    for dirpath in dirpaths:
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)


from types import SimpleNamespace
cfg = SimpleNamespace()

cfg.dpath_csvRoot = 'csv/'
cfg.dpath_dataRoot = 'data/'

cfg.dpath_mrxsRoot = '../CLAM/data0_mrxs/' #.mrxs WSI

cfg.dpath_patchRoot = os.path.join(cfg.dpath_dataRoot, 'data1_expatch/') #.h5
cfg.dpath_patchset = os.path.join(cfg.dpath_patchRoot, 'patches')
cfg.dpath_patchset_masks = os.path.join(cfg.dpath_patchRoot, 'masks')
cfg.dpath_patchset_stitch = os.path.join(cfg.dpath_patchRoot, 'stitches')

cfg.dpath_featuresRoot = os.path.join(cfg.dpath_dataRoot, 'data2_features/') #.h5 + .pt
cfg.dpath_features_pt = os.path.join(cfg.dpath_featuresRoot, 'pt_files/')
cfg.dpath_features_h5 = os.path.join(cfg.dpath_featuresRoot, 'h5_files/')

cfg.dpath_classtrainRoot = os.path.join(cfg.dpath_dataRoot, 'data3_classtrain/')
# ('--results_dir', default='./results', help='results directory (default: ./results)')
cfg.dpath_classtrain_results = os.path.join(cfg.dpath_classtrainRoot, 'results/')
# ('--split_dir', type=str, default=None, help='manually specify the set of splits to use, ' +'instead of infering from the task and label_frac argument (default: None)')
cfg.dpath_classtrain_splits = os.path.join(cfg.dpath_classtrainRoot, 'splits')
# ('--exp_code', type=str, help='experiment code for saving results')
cfg.dpath_classtrain_unknown = os.path.join(cfg.dpath_classtrainRoot, 'unknown')

cfg.dpath_milfolds = os.path.join(cfg.dpath_csvRoot, 'milFolds')



create_dirs(
    cfg.dpath_csvRoot,
    cfg.dpath_dataroot,
    cfg.dpath_patchRoot,
    cfg.dpath_patchset,
    cfg.dpath_patchset_masks,
    cfg.dpath_patchset_stitch,
    cfg.dpath_featuresRoot,
    cfg.dpath_features_pt,
    cfg.dpath_features_h5,
    cfg.dpath_classtrainRoot,
    cfg.dpath_classtrain_results,
    cfg.dpath_classtrain_splits,
    cfg.dpath_classtrain_unknown,
    cfg.dpath_milFolds,
)

cfg.fpath_map_patient_info = os.path.join(cfg.dpath_csvRoot, 'patient_info.csv')
cfg.fpath_map_patchset = os.path.join(cfg.dpath_csvRoot, 'map_patchset.csv')
cfg.fpath_map_classtrain = os.path.join(cfg.dpath_csvRoot, 'map_classtrain.csv')

cfg.fpath_fexmodel = 'pbo_model/pbo_res18.ckpt'
cfg.fexparam_batch_size=512
cfg.fexparam_patch_size=224
cfg.fexparam_slide_extension='.mrxs'
cfg.fexparam_no_auto_skip=False
