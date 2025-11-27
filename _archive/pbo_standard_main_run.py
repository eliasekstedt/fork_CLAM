
import os
import torch
import argparse
import numpy as np
import pandas as pd

from utils.file_utils import save_pkl
from utils.core_utils import train

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def do(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
        )
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train(
            datasets,
            i,
            args,
        )
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({
        'folds':folds,
        'test_auc':all_test_auc, 
        'val_auc':all_val_auc,
        'test_acc':all_test_acc,
        'val_acc':all_val_acc,
    })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--embed_dim', type=int, default=512)
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'pbo'], default='pbo')
parser.add_argument(
    '--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
    help='slide-level classification loss function (default: ce)')
parser.add_argument(
    '--data_root_dir', type=str, default='data/data2_features',
    help='data directory')
parser.add_argument(
    '--exp_code', type=str, default='exp_code_pbo',
    help='experiment code for saving results')
parser.add_argument(
    '--early_stopping', action='store_true', default=False,
    help='enable early stopping')
parser.add_argument(
    '--k', type=int, default=4,
    help='number of folds (default: 10)')
parser.add_argument(
    '--k_start', type=int, default=-1, 
    help='start fold (default: -1, last fold)')
parser.add_argument(
    '--k_end', type=int, default=-1,
    help='end fold (default: -1, first fold)')
parser.add_argument(
    '--log_data', action='store_true', default=False,
    help='log data using tensorboard')
parser.add_argument(
    '--lr', type=float, default=2e-4,
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--label_frac', type=float, default=1.0,
    help='fraction of training labels (default: 1.0)')
parser.add_argument(
    '--model_size', type=str, choices=['small', 'big'], default='small',
    help='size of model, does not affect mil')
parser.add_argument(
    '--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb',
    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument(
    '--max_epochs', type=int, default=200,
    help='maximum number of epochs to train (default: 200)')
parser.add_argument(
    '--reg', type=float, default=1e-5,
    help='weight decay (default: 1e-5)')
parser.add_argument(
    '--results_dir', default='clsStandardRoot/results',
    help='results directory (default: ./results)')
parser.add_argument(
    '--seed', type=int, default=7,
    help='random seed for reproducible experiment (default: 1)')
parser.add_argument(
    '--split_dir', type=str, default='splits',
    help='manually specify the set of splits to use, '
    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument(
    '--testing', action='store_true', default=False,
    help='debugging tool')
parser.add_argument(
    '--weighted_sample', action='store_true', default=False,
    help='enable weighted sampling')

### CLAM specific options
parser.add_argument(
    '--no_inst_cluster', action='store_true', default=False,
    help='disable instance-level clustering')
parser.add_argument(
    '--inst_loss', type=str, choices=['svm', 'ce', None], default='svm',
    help='instance-level clustering loss function (default: None)')
parser.add_argument(
    '--subtyping', action='store_true', default=False,
    help='subtyping problem')
parser.add_argument(
    '--bag_weight', type=float, default=0.7,
    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument(
    '--B', type=int, default=8,
    help='numbr of positive/negative patches to sample for clam')
args = parser.parse_args()


#fpath_src_csv = 'csv/map_classtrain.csv'

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_torch(args.seed)

encoding_size = 1024
settings = {
    'num_splits': args.k, 
    'k_start': args.k_start,
    'k_end': args.k_end,
    'task': args.task,
    'max_epochs': args.max_epochs, 
    'results_dir': args.results_dir, 
    'lr': args.lr,
    'experiment': args.exp_code,
    'reg': args.reg,
    'label_frac': args.label_frac,
    'bag_loss': args.bag_loss,
    'seed': args.seed,
    'model_type': args.model_type,
    'model_size': args.model_size,
    "use_drop_out": args.drop_out,
    'weighted_sample': args.weighted_sample,
    'opt': args.opt
}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

"""
import pandas as pd
def to_str_label(i_label):
    return ['normal_tissue', 'tumor_tissue'][i_label]
    
df = pd.read_csv(fpath_src_csv)
df['label'] = df['label'].apply(to_str_label)
df.to_csv(fpath_src_csv, index=False)
"""
fpath_src_csv = 'csv/map_classtrain_standard.csv'


args.n_classes = 2
from dataset_modules.dataset_generic import Generic_MIL_Dataset
dataset = Generic_MIL_Dataset(
    csv_path=fpath_src_csv,
    data_dir=args.data_root_dir,
    shuffle=False,
    seed=args.seed,
    print_info=True,
    label_dict={'normal_tissue':0, 'tumor_tissue':1},
    patient_strat=False,
    ignore=[],
)

if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

args.split_dir = 'clsStandardRoot/results/splits/'
if not os.path.exists(args.split_dir):
    os.makedirs(args.split_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)
#raise SystemExit

settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

do(args)


"""
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10  --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data
"""

