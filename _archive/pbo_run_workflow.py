
from _archive.pbo_config import *
do_assemble_patient_info = True
do_generate_patches = False
do_feature_extraction = False
do_foldsplitting = False
do_classifier_training = False

if __name__ == '__main__':
    if do_assemble_patient_info:
        pass


    if do_generate_patches:
        from coord_search.coord_search import PatchCoordMapper # coord finder
        patchset_generator = PatchCoordMapper(
            dpath_mrxs=cfg.dpath_mrxsRoot,
            dpath_patchRoot=cfg.dpath_patchRoot,
            dpath_patchset=cfg.dpath_patchcoords,
            dpath_patchset_masks=cfg.dpath_patchcoords_masks,
            dpath_patchset_stitch=cfg.dpath_patchcoords_stitch,
            mthresh=cfg.mthresh,
            close=cfg.close,
            a_t=cfg.a_t,
            a_h=cfg.a_h,
            max_holes=cfg.max_holes,
            live=False,
        )
        patchset_generator()

        # generate map1_mrxsSlides.csv
        from pbo_map_generators import PatchcoordMapGenerator
        PatchcoordMapGenerator(
            dpath_mrxs=cfg.dpath_mrxsRoot,
            dpath_patchset=cfg.dpath_patchset,
            fpath_map_patient=cfg.fpath_map_patient_info,
            fpath_map_patchset=cfg.fpath_map_patchcoords,
        )

    if do_feature_extraction:
        from pbo_extract_features import FeatureExtractor
        from _archive.patchQualVal import PatchFilter
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
            quality_filter=PatchFilter(),
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
        from MIL.trainer import MilTrainWrapper
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
points to better replicate the article:
* NOTE that patients and slides have already been excluded
up to the point |processed images| in figure 1 consort diagram

other areas:
* verify that the shape of the data that enters the mil model makes sense
* meaning of 'Setting tau to 1.0'
* architecture single_branch + small. perhaps i took some things for
granted when transcribing the model. e.g, gated or not? did the size get translated
properly? what about cross entropy bag loss as loss function?
"""

"""
potentially low quality coord extraction:
* 069
* 078
* 089
* 104
* 106
* 113
* 118
* 124
* 136
* 137
* 151_G
* 154
* 166

needle:
* 046, 075
"""