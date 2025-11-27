
import os
import argparse
import numpy as np
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, save_splits


parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument(
    '--label_frac', type=float, default= 1.0,
    help='fraction of labels (default: 1)')
parser.add_argument(
    '--seed', type=int, default=1,
    help='random seed (default: 1)')
parser.add_argument(
    '--k', type=int, default=4,
    help='number of splits (default: 10)')
parser.add_argument(
    '--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping', 'pbo'], default='pbo')
parser.add_argument(
    '--val_frac', type=float, default= 0.1,
    help='fraction of labels for validation (default: 0.1)')
parser.add_argument(
    '--test_frac', type=float, default= 0.1,
    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

fpath_src_csv = 'csv/map_classtrain_standard.csv'
args.n_classes=2
dataset = Generic_WSI_Classification_Dataset(
    csv_path=fpath_src_csv,
    shuffle = False, 
    seed = args.seed, 
    print_info = True,
    label_dict = {'normal_tissue':0, 'tumor_tissue':1},
    patient_strat=True,
    ignore=[]
)

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

split_dir = 'clsStandardRoot/results/splits'
os.makedirs(split_dir, exist_ok=True)
dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=args.label_frac)
for i in range(args.k):
    dataset.set_splits()
    descriptor_df = dataset.test_split_gen(return_descriptor=True)
    splits = dataset.return_splits(from_id=True)
    save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
    save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
    descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



