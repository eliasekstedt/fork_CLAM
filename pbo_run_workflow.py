
from pbo_config import *

do_generate_patches = True
do_feature_extraction = True
do_foldsplitting = True
do_classifier_training = True

if __name__ == '__main__':
    if do_generate_patches:
        from pbo_create_patchsets import PatchsetGenerator # coord finder
        patchset_generator = PatchsetGenerator(
            dpath_mrxs=cfg.dpath_mrxsRoot,
            dpath_patchRoot=cfg.dpath_patchRoot,
            dpath_patchset=cfg.dpath_patchset,
            dpath_patchset_masks=cfg.dpath_patchset_masks,
            dpath_patchset_stitch=cfg.dpath_patchset_stitch,
            mthresh=cfg.mthresh,
            close=cfg.close,
            a_t=cfg.a_t,
            a_h=cfg.a_h,
            max_holes=cfg.max_holes,
        )
        patchset_generator()

        # generate map1_mrxsSlides.csv
        from pbo_map_generators import PatchsetMapGenerator
        PatchsetMapGenerator(
            dpath_mrxs=cfg.dpath_mrxsRoot,
            dpath_patchset=cfg.dpath_patchset,
            fpath_map_patient=cfg.fpath_map_patient_info,
            fpath_map_patchset=cfg.fpath_map_patchset,
        )

    if do_feature_extraction:
        from pbo_extract_features import FeatureExtractor
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

"""
potential problems:
* does encoder return useful information
    - expected augmentations?
    - expected patch scale, i.e, number of nuclei that should fit on patch

"""


"""
should write down notes to expose what i dont understand about the workflow. for example:
* when is CLAM used in the workflow and why is it more suited than for example resnetX?
* what is clustering based on, i.e what are the attributes?
* when do attention mechanisms come in? feature extraction or after? how do attention mechanisms work?
"""