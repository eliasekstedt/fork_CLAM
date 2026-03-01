
from config import *

do_generate_coords = False
do_qualVal = False
do_patch_quality_check = False
do_feature_extraction = False
do_foldsplitting = True
do_mil = True

if __name__ == '__main__':
    if do_generate_coords:
        from find_coords import CoordGenerator
        cw = CoordGenerator(
            dpath_wsiRoot=cfg.dpath_wsiRoot,
            dpath_wsiCoordRoot=cfg.dpath_wsiCoordRoot,
            dpath_wsiCoord=cfg.dpath_wsiCoord,
            dpath_mask=cfg.dpath_mask,
            dpath_stitch=cfg.dpath_stitch,
            fpath_segmParam=cfg.fpath_segmParam,
            fpath_segmlog=cfg.fpath_segmlog,
        )
        cw()

    if do_qualVal:
        from patch_meta import MetaLooper
        MetaLooper(
            dpath_wsiRoot=cfg.dpath_wsiRoot,
            dpath_qualityLog=cfg.dpath_qualityLog,
            dpath_wsiCoords=cfg.dpath_wsiCoords,
            fpath_segmlog=cfg.fpath_segmlog,
        )

    if do_patch_quality_check:
        from patch_quality_check import QualityVisualizer
        QualityVisualizer(
            dpath_wsiRoot=cfg.dpath_wsiRoot,
            dpath_wsiCoords=cfg.dpath_wsiCoords,
            dpath_qualityLog=cfg.dpath_qualityLog,
            fltr_params=cfg.fltr_params,
            dpath_geometryCheck=cfg.dpath_geometryCheck,
            dpath_keepVreject=cfg.dpath_keepVreject,
            fpath_patchProperties=cfg.fpath_patchProperties,
            fpath_perSlideInfo=cfg.fpath_perSlideInfo,
        )

    if do_feature_extraction:
        from encode_patches import FeatureX
        FeatureX(
            dpath_qualityLog=cfg.dpath_qualityLog,
            dpath_wsiRoot=cfg.dpath_wsiRoot,
            dpath_ptFeature=cfg.dpath_ptFeature,
            dpath_h5Feature=cfg.dpath_h5Feature,
            fpath_segmlog=cfg.fpath_segmlog,
            fpath_encodingMap=cfg.fpath_encodingMap,
            fpath_Xmodel=cfg.fpath_Xmodel,
            fpath_patientInfo=cfg.fpath_patientInfo,
            batch_size=cfg.X_batch_size,
            augm=cfg.Xaugm,
            fltr_params=cfg.fltr_params,
            target_bag_size=cfg.target_bag_size,
            dpath_sampleFltrpassed=cfg.dpath_sampleFltrpassed,
        )

    import random
    import string
    while True:

        rid = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        if do_foldsplitting:
            from foldsplitter import FeatureMapSplitter
            FeatureMapSplitter(
                fpath_encodingMap=cfg.fpath_encodingMap,
                excl_by_id=cfg.excl_by_id,
                fpath_fold0=cfg.fpath_fold0,
                fpath_fold1=cfg.fpath_fold1,
                rid=rid,
            )

        if do_mil:
            from pbo_mil_trainer import MilTrainWrapper

            if cfg.state_dict == 'history': # if true, select most recent model
                with open(f"run/history/history.txt", 'r') as file:
                    fpath_state_dict = file.readlines()[-1]
                    while not fpath_state_dict.endswith('model.pth'):
                        fpath_state_dict = fpath_state_dict[:-1]
            else:
                fpath_state_dict = cfg.state_dict

            hparam = cfg.hparam
            hparam['rid'] = str(rid)
            MilTrainWrapper(
                dpath_ptFeature=cfg.dpath_ptFeature,
                fpath_fold0=cfg.fpath_fold0,
                fpath_fold1=cfg.fpath_fold1,
                hparam=hparam,
                fpath_state_dict=fpath_state_dict,
                tag=cfg.tag,
                device='cuda:0',
            )

"""
TcNJrW - 01_15_03_55
syTKD1 - 01_15_35_08
HVqfdo - 01_15_46_24
e94S5n - 01_16_43_19
WVjqmE - 01_16_01_00
z9wKz5
IjQCuT - 01_14_46_33
wm1e2p - 01_17_17_26
qxiUbc - 
YDESRs - 

gRv8Oa
49pTVk
kng2kA
WONmXO
azsWfm
ZkQI5i
"""