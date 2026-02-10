
from config import *

do_generate_coords = False
do_qualVal = True
do_feature_extraction = False
do_foldsplitting = False
do_classifier_training = False

if __name__ == '__main__':
    if do_generate_coords:
        from config import *
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
        from quality_measure import QALooper
        QALooper(
            dpath_wsiRoot=cfg.dpath_wsiRoot,
            dpath_qualityLog=cfg.dpath_qualityLog,
            dpath_wsiCoords=cfg.dpath_wsiCoords,
            fpath_segmlog=cfg.fpath_segmlog,
        )

    if do_feature_extraction:
        from extract_features import FeatureExtractor
        feature_extractor = FeatureExtractor(
            dpath_patchset=cfg.dpath_patchset,
            dpath_mrxsRoot=cfg.dpath_mrxsRoot,
            dpath_features_pt=cfg.dpath_features_pt,
            dpath_features_h5=cfg.dpath_features_h5,
            fpath_map_patchset=cfg.fpath_map_patchset,
            fpath_model=cfg.fpath_Xmodel,
            batch_size=cfg.fexparam_batch_size,
            patch_size=cfg.fexparam_patch_size,
            slide_extension=cfg.fexparam_slide_extension,
            no_auto_skip=cfg.fexparam_no_auto_skip,
        )
        feature_extractor()










    if do_foldsplitting:
        from pbo_map_generators import ClassifierMapGenerator
        fold_splitter = ClassifierMapGenerator(
            fpath_map_patchset=cfg.fpath_map_patchset,
            fpath_map_fold_0=cfg.fpath_map_fold_0,
            fpath_map_fold_1=cfg.fpath_map_fold_1,
        )

    if do_classifier_training:
        from pbo_mil_trainer import MilTrainWrapper
        """
        why does the train epoch factor in instance loss in loss calculation but not val epoch?
        """
        if cfg.state_dict == 'history': # if true, select most recent model
            with open(f"run/history/history.txt", 'r') as file:
                fpath_state_dict = file.readlines()[-1]
                while not fpath_state_dict.endswith('model.pth'):
                    fpath_state_dict = fpath_state_dict[:-1]
        else:
            fpath_state_dict = cfg.state_dict

        wrapper = MilTrainWrapper(
            dpath_features_pt=cfg.dpath_features_pt,
            fpath_map_fold_0=cfg.fpath_map_fold_0,
            fpath_map_fold_1=cfg.fpath_map_fold_1,
            hparam=cfg.hparam,
            fpath_state_dict=fpath_state_dict,
            augm=cfg.augm,
            tag=cfg.tag,
            device='cuda:0',
        )

