
from config import *

def compute_score(vec):
    vec = [[0, item][int(item > 0.5)] for item in vec]
    score = 0
    for i in range(3, len(vec)+1):
        window = vec[i-3:i]
        score += np.prod(window) / len(vec)
    return score

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

    
    from foldsplitter import FeatureMapSplitter
    from pbo_mil_trainer import MilTrainWrapper
    while True:

        excl_const = [
            '037_DEF', '044_ABC', '048_G', '065_DEF', '073_ABC', '081_ABC',
            '085_CDE', '091_DE', '099_GHK', '144_AB', '151_AB', '154_ABC',
            '205_ABC', '209_KL', '099_ABC', '116_AB', '003_GH',
        ]
        
        excl_rnd_set = [
            '001_ABC', '002_ABC', '002_DEF', '004_GHK', '007_DEF', '010_KL',
            '012_AB', '014_CD', '016_DEF', '016_GH', '016_KL', '017_DEF',
            '019_ABC', '020_AB', '026_DEF', '028_EF', '030_GH', '032_AB',
            '032_EF', '033_KL', '034_EF', '036_AB', '038_DE', '039_DEF',
            '041_ABC', '042_DE', '045_DEF', '054_GH', '062_ABC', '066_GH',
            '068_ABC', '068_DEF', '071_FGH', '072_ABC', '074_ABC', '077_DEF',
            '078_CD', '080_ABC', '082_ABC', '083_DEF', '091_AB', '097_ABC',
            '102_AB', '108_CD', '109_ABC', '110_DEF', '111_EF', '115_DEF',
            '118_ABC', '118_DEF', '119_L', '120_GH', '125_DEF', '129_DEF',
            '132_DEF', '135_ABC', '136_FG', '140_DEF', '141_EF', '144_CD',
            '144_EF', '145_LMN', '148_ABC', '148_DEF', '150_DEF', '152_DEF',
            '155_DEF', '158_CD', '160_A', '160_C', '167_CD', '169_DEF', '172_ABC',
            '172_KL', '173_FGH', '176_GHK', '177_DEF', '181_AB', '181_G', '183_DEF',
            '185_ABC', '189_EF', '198_EF', '205_DEF', '208_DEF', '212_ABC',
            '123_EF', '151_CD', '171_M', '197_ABC', '201_KL', '124_AB', '104_AB',
            '200_AB', '025_AB', '027_FG', '031_ABC', '049_DEF', '051_GH', '055_ABC',
            '074_DEF', '080_G', '107_ABC', '107_L', '117_KL', '143_F', '146_DEF',
            '162_DEF', '163_ABC', '168_DEF', '193_ABCD', '203_ABC', '208_ABC',
            '210_L', '213_LMN', '164_DEF',
        ]

        #incl_always = ['125_DEF', '155_DEF', '016_KL', '158_CD', '080_ABC', '010_KL', '041_ABC', '102_AB', '201_KL', '002_ABC', '020_AB']
        #excl_rnd = [el for el in excl_rnd if not el in incl_always]
        #assert all([not el in incl_always for el in excl_by_id])


        import random as rnd
        rnd.shuffle(excl_rnd_set)
        wedge = int(len(excl_rnd_set) * 0.04)

        excl_rnd = excl_rnd_set[wedge:]
        incl_rnd = excl_rnd_set[:wedge]
        excl_by_id = excl_const + excl_rnd
        assert all([not el in excl_rnd for el in incl_rnd])
        #raise SystemExit
        
        
        import numpy as np
        import pandas as pd
        import string
        rid = ''.join(rnd.choices(string.ascii_letters + string.digits, k=6))
        
        for _ in range(5):
            FeatureMapSplitter(
                fpath_encodingMap=cfg.fpath_encodingMap,
                excl_by_id=excl_by_id,
                fpath_fold0=cfg.fpath_fold0,
                fpath_fold1=cfg.fpath_fold1,
            )
            incl_rnd_fold_0 = [
                slide_id for slide_id in incl_rnd
                if slide_id in pd.read_csv(cfg.fpath_fold0)['slide_id'].to_list()
            ]
            print(incl_rnd_fold_0)
            if len(incl_rnd_fold_0) == 0:
                print(f'incl_rnd_fold_0 is empty: {incl_rnd_fold_0}')
                break

            if cfg.state_dict == 'history': # if true, select most recent model
                with open(f"run/history/history.txt", 'r') as file:
                    fpath_state_dict = file.readlines()[-1]
                    while not fpath_state_dict.endswith('model.pth'):
                        fpath_state_dict = fpath_state_dict[:-1]
            else:
                fpath_state_dict = cfg.state_dict


            wrapper = MilTrainWrapper(
                dpath_ptFeature=cfg.dpath_ptFeature,
                fpath_fold0=cfg.fpath_fold0,
                fpath_fold1=cfg.fpath_fold1,
                hparam=cfg.hparam,
                fpath_state_dict=fpath_state_dict,
                tag=cfg.tag,
                device='cuda:0',
            )

            ########################
            ########################
            
            #selection_score = np.sum([item > 0.5 for item in wrapper.trainer.val_precision]) / cfg.hparam['nr_epochs']
            score = compute_score(wrapper.trainer.val_precision)
            fpath_selectionScores = Path('ss_fold_specified.csv')
            row = pd.DataFrame({
                'rid':rid,
                'score':[-1, score][int(score==0)],
                'incl_of_fold0':[incl_rnd_fold_0],
            })
            if not fpath_selectionScores.is_file():
                ss_df = row
            else:
                ss_df = pd.read_csv(fpath_selectionScores)
                ss_df = pd.concat([ss_df, row], axis=0)

            ss_df.to_csv(fpath_selectionScores, index=False)
            print(f"score: {score}\n")
            if score >= 0:
                break
            ########################
            ########################

"""
the best:
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

just okay:
gRv8Oa
49pTVk
kng2kA
WONmXO
azsWfm
ZkQI5i
qJcPcu
AXhV7i


#import random
#import string
#rid = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
"""