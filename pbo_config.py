
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

cfg.dpath_classifierRoot = os.path.join(cfg.dpath_dataRoot, 'data3_classifier')
cfg.dpath_milFolds = os.path.join(cfg.dpath_csvRoot, 'milFolds')

create_dirs(
    cfg.dpath_csvRoot,
    cfg.dpath_dataRoot,
    cfg.dpath_patchRoot,
    cfg.dpath_patchset,
    cfg.dpath_patchset_masks,
    cfg.dpath_patchset_stitch,
    cfg.dpath_featuresRoot,
    cfg.dpath_features_pt,
    cfg.dpath_features_h5,
    cfg.dpath_classifierRoot,
    cfg.dpath_milFolds,
)


cfg.fpath_map_patient_info = os.path.join(cfg.dpath_csvRoot, 'patient_info.csv')
cfg.fpath_map_patchset = os.path.join(cfg.dpath_csvRoot, 'map_patchset.csv')
cfg.fpath_map_fold_0 = os.path.join(cfg.dpath_milFolds, 'map_fold_0.csv')
cfg.fpath_map_fold_1 = os.path.join(cfg.dpath_milFolds, 'map_fold_1.csv')

cfg.fpath_fexmodel = os.path.join(cfg.dpath_dataRoot, 'pbo_fexmodel.ckpt')




cfg.fexparam_batch_size=512
cfg.fexparam_patch_size=224
cfg.fexparam_slide_extension='.mrxs'
cfg.fexparam_no_auto_skip=False

cfg.tag = 'unspecified'
cfg.hparam = {
    'dropout': 0.0, #0.25,
    'batch_size':1,
    'learning_rate':0.01,
    'weight_decay':1e-6,
    'nr_epochs':200,
}
cfg.augm = {
    'placeholder':True
}