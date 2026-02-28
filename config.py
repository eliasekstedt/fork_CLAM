
from pathlib import Path

def create_dirs(*dirpaths):
    for dirpath in dirpaths:
        dirpath.mkdir(exist_ok=True)

wsi_origin = '../CLAM/data0_mrxs/'


from types import SimpleNamespace
cfg = SimpleNamespace()
cfg.dpath_wsiRoot = Path(wsi_origin)

cfg.dpath_csvRoot = Path('csv/')
cfg.dpath_qualityLog = cfg.dpath_csvRoot / 'quality_log'
cfg.dpath_milFolds = cfg.dpath_csvRoot / 'milFolds'


cfg.dpath_dataRoot = Path('data/')

cfg.dpath_wsiCoordRoot = cfg.dpath_dataRoot / '0_wsiCoordRoot'
cfg.dpath_wsiCoords = cfg.dpath_wsiCoordRoot / 'coord'
cfg.dpath_mask = cfg.dpath_wsiCoordRoot / 'mask'
cfg.dpath_stitch = cfg.dpath_wsiCoordRoot / 'stitch'

cfg.dpath_featureRoot = cfg.dpath_dataRoot / '1_featureRoot'
cfg.dpath_ptFeature = cfg.dpath_featureRoot / 'ptFeature'
cfg.dpath_h5Feature = cfg.dpath_featureRoot / 'h5Feature'

cfg.dpath_diagnosticsRoot = Path('diagnostics')
cfg.dpath_patchControl = cfg.dpath_diagnosticsRoot / 'patch_control'
cfg.dpath_geometryCheck = cfg.dpath_patchControl / 'geometry_check'
cfg.dpath_keepVreject = cfg.dpath_patchControl / 'keepVreject'
cfg.dpath_sampleFltrpassed = cfg.dpath_diagnosticsRoot / 'fltrpassed_samples'

create_dirs(
    cfg.dpath_csvRoot,
    cfg.dpath_dataRoot,
    cfg.dpath_qualityLog,
    cfg.dpath_wsiCoordRoot,
    cfg.dpath_wsiCoords,
    cfg.dpath_mask,
    cfg.dpath_stitch,
    cfg.dpath_featureRoot,
    cfg.dpath_ptFeature,
    cfg.dpath_h5Feature,
    cfg.dpath_milFolds,
    cfg.dpath_diagnosticsRoot,
    cfg.dpath_patchControl,
    cfg.dpath_geometryCheck,
    cfg.dpath_keepVreject,
    cfg.dpath_sampleFltrpassed,
)

cfg.fpath_patientInfo = cfg.dpath_csvRoot / 'patient_info.csv'
cfg.fpath_segmParam = cfg.dpath_csvRoot / 'segmParams.csv'
cfg.fpath_segmlog = cfg.dpath_csvRoot / 'segmlog.csv'
cfg.fpath_encodingMap = cfg.dpath_csvRoot / 'encoding_map.csv'
cfg.fpath_patchProperties = cfg.dpath_patchControl / 'patchProperties.png'
cfg.fpath_perSlideInfo = cfg.dpath_patchControl / 'per_slide_info.csv'
cfg.fpath_fold0 = cfg.dpath_milFolds / 'fold0.csv'
cfg.fpath_fold1 = cfg.dpath_milFolds / 'fold1.csv'

cfg.fltr_params = {
    'bg':0.5,
    'blur':40,
}

# feauture extraction
cfg.fpath_Xmodel = cfg.dpath_dataRoot / 'Xmodel.ckpt'

cfg.X_batch_size = 512 # best kept small for appropriate randomization, aka even bag size per slide
cfg.X_patch_size = 224
cfg.target_bag_size = 2500

cfg.Xaugm = {
    'size':224,
    'znorm_mean':[0.485, 0.456, 0.406],
    'znorm_std':[0.229, 0.224, 0.225],
}


cfg.tag = 'IN_confirmed'
cfg.hparam = {
    'dropout':0.25,
    'batch_size':1,
    'learning_rate':1e-4,
    'weight_decay':5e-3,
    'nr_epochs':300,
}

cfg.state_dict = ''