
from pbo_config import *

do_generate_patches = False
do_feature_extraction = False
do_foldsplitting = False
do_classifier_training = True


if do_generate_patches:
    """
    basic fully automated run - aka, extract patches from slides
    python create_patches_fp.py
    --source DATA_DIRECTORY #data0_MRXS
    --save_dir RESULTS_DIRECTORY #data1_expatch
    --patch_size 256 --seg --patch --stitch 


    note: figure out where to convert slides to HSV color space

    """

    from pbo_create_patchsets import PatchsetGenerator
    patchset_generator = PatchsetGenerator(
        dpath_mrxs=cfg.dpath_mrxsRoot,
        dpath_patchRoot=cfg.dpath_patchRoot,
        dpath_patchset=cfg.dpath_patchset,
        dpath_patchset_masks=cfg.dpath_patchset_masks,
        dpath_patchset_stitch=cfg.dpath_patchset_stitch,
    )
    patchset_generator()

    """
    generate map1_mrxsSlides.csv
    """
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
        fpath_model=cfg.fpath_fexmodel,
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
should write down notes to expose what i dont understand about the workflow. for example:
* when is CLAM used in the workflow and why is it more suited than for example resnetX?
* what is clustering based on, i.e what are the attributes?
* when do attention mechanisms come in? feature extraction or after? how do attention mechanisms work?

"""


"""
if do_classifier_training:
    if False:
        from pbo.map_generators import ClasstrainMapGenerator
        ClasstrainMapGenerator(
            fpath_map_patchset=cfg.fpath_map_patchset,
            fpath_map_classtrain=cfg.fpath_map_classtrain,
        )

    from pbo_classification_training import PBOGenericMILDataset
    PBOGenericMILDataset(
        fpath_map_classtrain=cfg.fpath_map_classtrain,
        dpath_features_pt=cfg.dpath_features_pt,
        shuffle=False, 
        seed=1, 
        print_info=True,
        patient_strat=False,
        ignore=[]
    )
"""

"""
optionally set pretrained encoder - for prosBiOps thats the res18
"""
"""
feature Extraction (gpu example) - aka, patches to .pt encodings
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py
--data_h5_dir DIR_TO_COORDS #data1_expatch
--data_slide_dir DATA_DIRECTORY #data0_mrxs
--csv_path CSV_FILE_NAME #cfg.fpath_map_mrsxSlides
--feat_dir FEATURES_DIRECTORY #cfg.dpath_vectors
--batch_size 512 --slide_ext .svs
** note also parameter model_name
"""
"""
set training splits
python create_splits_seq.py
--task task_1_tumor_vs_normal
--seed 1
--k 10
"""
"""
GPU Training Example for Binary Positive vs. Negative Classification
CUDA_VISIBLE_DEVICES=0 python main.py
--drop_out 0.25 --early_stopping --lr 2e-4 --k 10 
--exp_code task_1_tumor_vs_normal_CLAM_50 # outdir, aka classtrain
--weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data
--data_root_dir DATA_ROOT_DIR # cfg.dpath_vectors
--embed_dim 1024
"""
"""
evaluation
CUDA_VISIBLE_DEVICES=0 python eval.py --k 10 --models_exp_code task_1_tumor_vs_normal_CLAM_50_s1 --save_exp_code task_1_tumor_vs_normal_CLAM_50_s1_cv --task task_1_tumor_vs_normal --model_type clam_sb --results_dir results --data_root_dir DATA_ROOT_DIR --embed_dim 1024
"""
"""
heatmap visualisation
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config config_template.yaml


“(1) Can we get insight into the role of surface protein density and the role of different structural proteins by comparing the fusion kinetics of different particles.
(2) what can we learn about the impact of the influenza matrix protein on membrane fusion with generated virus-like particles as specified.”

"""


# LEARN TO UNDERSTAND EACH INPUT PARAMETER. BUILD A TABLE WHERE THE COLUMNS ARE FILES, THE ROWS ARE INPUT PARAMETER NAME AND THE ELEMENTS ARE PARAMETER DESCRIPTION
# ALSO, COMPLETE THE CFG, and remember to add the relevant directories to .gitignore


