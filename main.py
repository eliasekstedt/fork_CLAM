
from config import *

do_generate_coords = False
do_qualVal = False
do_patch_quality_check = False
do_feature_extraction = False
do_foldsplitting = True
do_mil = True

if __name__ == '__main__':
    if do_generate_coords:
        """
        For each slide, generate .h5 files containing coords
        and metadata for patches useful in subsequent patch
        extraction.
        """
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
        """
        For each slide, do quality measurements for each patch.
        Generate a .csv file where quality measurements for each
        patch are stored. Useful for downstream filtering adjustments.
        """
        from patch_meta import MetaLooper
        MetaLooper(
            dpath_wsiRoot=cfg.dpath_wsiRoot,
            dpath_qualityLog=cfg.dpath_qualityLog,
            dpath_wsiCoords=cfg.dpath_wsiCoords,
            fpath_segmlog=cfg.fpath_segmlog,
        )

    if do_patch_quality_check:
        """
        Produces files for verification of propper patch extraction,
        e.g no patch overlap or missing space and samples to show
        rejected vs accepted patches given set filtering parameters.
        """
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
        """
        For each slide, processes all patches that pass given the
        filtering params, for each patch generates feature vectors
        and collects those feature vectors in a bag to represent
        the whole slide.
        """
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

    if do_foldsplitting:
        """
        Splits the slides into training and validation data. this
        is done in a way that balances patient properties like age
        and psa.
        """
        from foldsplitter import FeatureMapSplitter
        FeatureMapSplitter(
            fpath_encodingMap=cfg.fpath_encodingMap,
            fpath_fold0=cfg.fpath_fold0,
            fpath_fold1=cfg.fpath_fold1,
        )
    
    if do_mil:
        """
        Trains the CLAM-classifier on the bags of feature vectors
        that represents the slides.
        """
        if cfg.state_dict == 'history': # if true, select most recent model
            with open(f"run/history/history.txt", 'r') as file:
                fpath_state_dict = file.readlines()[-1]
                while not fpath_state_dict.endswith('model.pth'):
                    fpath_state_dict = fpath_state_dict[:-1]
        else:
            fpath_state_dict = cfg.state_dict

        from pbo_mil_trainer import MilTrainWrapper
        wrapper = MilTrainWrapper(
            dpath_ptFeature=cfg.dpath_ptFeature,
            fpath_fold0=cfg.fpath_fold0,
            fpath_fold1=cfg.fpath_fold1,
            hparam=cfg.hparam,
            fpath_state_dict=fpath_state_dict,
            tag=cfg.tag,
            device='cuda:0',
        )
