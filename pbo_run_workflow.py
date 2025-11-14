
from pbo.pbo_config import *

do_generate_patches = True # .mrxs -> .h5 + create patchset map

if do_generate_patches:
    from pbo.create_patchsets import PatchsetGenerator
    patchset_generator = PatchsetGenerator(
        dpath_mrxs=cfg.dpath_mrxs,
        dpath_patchset=cfg.dpath_patchset,
    )
    patchset_generator()

    from pbo.map_generators import PatchsetMapGenerator
    PatchsetMapGenerator(
        dpath_mrxs=cfg.dpath_mrxs,
        dpath_patchset=cfg.dpath_patchset,
        fpath_map_patient=cfg.fpath_map_patient_info,
        fpath_map_patchset=cfg.fpath_map_patchset,
    )

"""
basic fully automated run - aka, extract patches from slides
python create_patches_fp.py
--source DATA_DIRECTORY #data0_MRXS
--save_dir RESULTS_DIRECTORY #data1_expatch
--patch_size 256 --seg --patch --stitch 
"""

"""
generate map1_mrxsSlides.csv
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
--drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data
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
"""


# LEARN TO UNDERSTAND EACH INPUT PARAMETER. BUILD A TABLE WHERE THE COLUMNS ARE FILES, THE ROWS ARE INPUT PARAMETER NAME AND THE ELEMENTS ARE PARAMETER DESCRIPTION
# ALSO, COMPLETE THE CFG, and remember to add the relevant directories to .gitignore


