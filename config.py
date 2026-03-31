
from pathlib import Path
from types import SimpleNamespace
cfg = SimpleNamespace()

def create_dirs(*dirpaths):
    for dirpath in dirpaths:
        dirpath.mkdir(exist_ok=True)

### only paths need specifying ###

wsi_origin = '../CLAM/data0_mrxs/'

######### define steps ###########
cfg.do_coord_search = False
cfg.do_quality_check = False
cfg.do_patch_quality_vis = False
cfg.do_feature_extraction = False
cfg.do_foldsplitting = False
cfg.do_mil = False
cfg.do_crossval = False
##################################

cfg.dpath_wsiRoot = Path(wsi_origin)
cfg.dpath_csvRoot = Path('csv/')
cfg.dpath_qualityLog = cfg.dpath_csvRoot / 'quality_log'
cfg.dpath_milfolds = cfg.dpath_csvRoot / 'milFolds'
cfg.holdout_fold_name = 'fold_0.csv'

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
    cfg.dpath_milfolds,
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

from datetime import datetime
current = datetime.now()

n_epochs = 3
tag_name = 'unspecified_run'
cfg.tag = f'{tag_name}' + '_{}_{}_{}_{}'.format(
    str(current)[8:10], str(current)[11:13],
    str(current)[14:16], str(current)[17:19],
)

cfg.hparam = {
    'dropout':0.5,
    'batch_size':1,
    'learning_rate':1e-3,
    'weight_decay':5e-5,
    'nr_epochs':30,
}

cfg.state_dict = ''
if cfg.state_dict == 'history': # if true, select most recent model
    with open(f"run/history/history.txt", 'r') as file:
        fpath_state_dict = file.readlines()[-1]
        while not fpath_state_dict.endswith('model.pth'):
            fpath_state_dict = fpath_state_dict[:-1]
else:
    fpath_state_dict = cfg.state_dict