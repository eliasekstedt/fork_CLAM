
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

cfg.excl_by_id = ['037_DEF', '044_ABC', '048_G', '065_DEF', '073_ABC', '081_ABC', '085_CDE', '091_DE', '099_GHK', '144_AB', '151_AB', '154_ABC', '205_ABC', '209_KL', '099_ABC', '116_AB', '003_GH']
#TcNJrW: cfg.excl_by_id = ['001_DEF', '003_GH', '004_ABC', '008_DEF', '008_LMN', '010_DEF', '010_GH', '010_KL', '016_DEF', '018_ABC', '020_AB', '020_CD', '022_AB', '024_DEF', '025_EF', '026_ABC', '028_EF', '031_DE', '032_EF', '036_AB', '037_A', '037_DEF', '043_AB', '044_ABC', '045_DEF', '048_G', '049_ABC', '051_DEF', '054_GH', '059_AB', '061_DEF', '063_ABC', '065_ABC', '065_DEF', '066_GH', '070_LMN', '073_ABC', '077_KL', '078_AB', '078_CD', '081_ABC', '083_ABC', '083_DEF', '086_CD', '088_ABC', '089_EF', '090_DEF', '090_GH', '091_DE', '092_ABC', '092_DEF', '099_ABC', '099_GHK', '108_CD', '110_DEF', '111_AB', '113_ABC', '115_DEF', '122_DEF', '124_EF', '125_ABC', '130_DEF', '131_EF', '133_ABC', '134_FGH', '135_DEF', '142_DEF', '143_DE', '144_AB', '145_DEF', '146_ABC', '150_ABC', '150_DEF', '151_AB', '158_CD', '159_DEF', '160_B', '160_R', '161_ABC', '165_CDE', '171_D', '171_F', '172_ABC', '174_ABC', '175_DEF', '179_ABC', '181_CD', '187_ABC', '189_AB', '189_EF', '190_DEF', '194_ABC', '196_DE', '199_ABC', '204_DEF', '205_ABC', '205_KL', '209_KL', '102_EF', '174_DEF', '213_GHK', '114_CD', '005_GH', '089_CD', '124_AB', '154_ABC', '178_KL', '031_ABC', '048_ABC', '050_DEF', '054_DEF', '055_DEF', '057_ABC', '062_DEF', '070_ABC', '074_DEF', '085_CDE', '095_DEF', '107_DEF', '108_AB', '116_AB', '133_DEF', '155_ABC', '163_ABC', '164_ABC', '193_ABCD', '198_CD', '201_ABC', '202_DEF', '203_ABC', '206_EF', '208_ABC', '182_ABC']

cfg.tag = 'excl_logging'
"""
cfg.hparam = {
    'dropout':0.25,
    'batch_size':1,
    'learning_rate':1e-4,
    'weight_decay':5e-3,
    'nr_epochs':120,
}
"""
cfg.hparam = {
    'dropout':0.5,
    'batch_size':1,
    'learning_rate':1e-3,
    'weight_decay':5e-5,
    'nr_epochs':60,
}

cfg.state_dict = 'run/excl2/01_13_21_25/model.pth'