
from config import *
#################################
from zpecial import *
#################################

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
    import string
    import random as rnd
    import pandas as pd
    #import matplotlib.pyplot as plt
    p = 0.5
    
    fpath_selectionScores = Path('ss_123.csv')
    dpath_scorePlots = Path('score_plots')
    dpath_scorePlots.mkdir(exist_ok=True)

    emap = pd.read_csv(cfg.fpath_encodingMap)

    while True:
        if fpath_selectionScores.is_file():
            instances = pd.read_csv(fpath_selectionScores)
        else:
            instances = pd.DataFrame({
                'rid':[],
                'score0':[],
                'score1':[],
                'score2':[],
                'label0_ratio':[],
                'slides_of_fold0':[],
            })

        #print(
        #    len(emap[emap['label'] == 0].sample(frac=p)['slide_id'].to_list()),
        #    len(emap[emap['label'] == 1].sample(frac=p)['slide_id'].to_list()),
        #); raise SystemExit
        excl_by_id = [# by visual inspection
            '037_DEF', '044_ABC', '048_G', '065_DEF', '073_ABC', '081_ABC',
            '085_CDE', '091_DE', '099_GHK', '144_AB', '151_AB', '154_ABC',
            '205_ABC', '209_KL', '099_ABC', '116_AB', '003_GH',
        ] + [# probably weak signal through testing
            '091_AB', '039_DEF', '125_DEF', '042_DE', '071_FGH', '167_CD',
            '200_AB', '162_DEF', '168_DEF', '001_ABC', '205_DEF'
        ] + get_variable_excl(instances, emap)
        #emap[emap['label'] == 0].sample(frac=p)['slide_id'].to_list() + emap[emap['label'] == 1].sample(frac=p)['slide_id'].to_list()

        label0_ratio = FeatureMapSplitter(
            fpath_encodingMap=cfg.fpath_encodingMap,
            excl_by_id=excl_by_id,
            fpath_fold0=cfg.fpath_fold0,
            fpath_fold1=cfg.fpath_fold1,
        ).label0_ratio
        ids_fold_0 = pd.read_csv(
            cfg.fpath_fold0
        ).sort_values(
            by='slide_id'
        )['slide_id'].to_list()
        
        #print(ids_fold_0, len(ids_fold_0))
        if len(ids_fold_0) == 0:
            print(f'ids_fold_0 is empty: {ids_fold_0}')
            break
        assert all([not id in excl_by_id for id in ids_fold_0])
        #raise SystemExit

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
        vprecision = wrapper.trainer.val_precision
        score0 = compute_score0(vprecision)
        score2 = compute_score2(vprecision)
        tcost = wrapper.trainer.traincost
        vcost = wrapper.trainer.valcost
        score1 = compute_score1(tcost, vcost)
        rid = ''.join(rnd.choices(string.ascii_letters + string.digits, k=6))
        row = pd.DataFrame({
            'rid':rid,
            'score0':score0,
            'score1':score1,
            'score2':score2,
            'label0_ratio':label0_ratio,
            'slides_of_fold0':[ids_fold_0],
        })
        if not fpath_selectionScores.is_file():
            ss_df = row
        else:
            ss_df = pd.read_csv(fpath_selectionScores)
            ss_df = pd.concat([ss_df, row], axis=0)

        ss_df.to_csv(fpath_selectionScores, index=False)
        print(f"scores:\n{score0},\n{score1},\n{score2}\n")
        print(f'nr_datapoints in {fpath_selectionScores.name}: {ss_df.shape[0]}')

        """
        plt.plot(tcost)
        plt.plot(vcost)
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(dpath_scorePlots / f"cost_{str(score1).replace('.', '')}_{rid}.png")
        plt.figure()

        tacc = wrapper.trainer.trainperformance
        vacc = wrapper.trainer.valperformance
        plt.plot(tacc)
        plt.plot(vacc)
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(dpath_scorePlots / f"acc_{str(score).replace('.', '')}_{rid}.png")
        plt.figure()
        """
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