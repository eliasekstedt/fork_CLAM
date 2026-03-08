
from config import *

def get_slide_ids(string):
    return string.replace(
        '[', ''
    ).replace(
        ']',''
    ).replace(
        "'",''
    ).replace(
        ' ',''
    ).split(',')

def testA():
    import pandas as pd
    import re
    import numpy as np
    yes = False

    class BadMapGenerator:
        def __init__(self, fpath_map_patchset, fpath_map_fold_0, fpath_map_fold_1):
            mmap = pd.read_csv(fpath_map_patchset)
            mmap['case_id'] = mmap['patchset_id'].apply(self.get_case_id)
            fold_0, fold_1 = self.split_kfold(mmap)
            if yes:
                fold_0.to_csv(fpath_map_fold_0, index=False)
                fold_1.to_csv(fpath_map_fold_1, index=False)
            else:
                """
                tdf = pd.concat([fold_0, fold_1], axis=0)
                print(tdf['age'].unique())
                tdf.loc[tdf['age']=='<55', 'age'] = 0
                tdf.loc[tdf['age']=='56-60', 'age'] = 1
                tdf.loc[tdf['age']=='61-65', 'age'] = 2
                tdf.loc[tdf['age']=='66-70', 'age'] = 3
                tdf.loc[tdf['age']=='71-75', 'age'] = 4
                tdf.loc[tdf['age']=='>75', 'age'] = 5
                print(tdf)
                #raise SystemExit
                import seaborn as sns
                import matplotlib.pyplot as plt
                sns.pairplot(tdf, hue='label')
                plt.show()
                """
                print(fold_0)
                print(fold_1)
            fold_0_label_0_ratio = 1 - (fold_0['label'].sum() / fold_0.shape[0])
            print(f"label 0 ratio: {fold_0_label_0_ratio}")

        def get_case_id(self, patchset_id):
            return re.match(r"(patient_[^_]+)", patchset_id).group(1)
        
        def split_kfold(self, mmap, k=5):
            def score_fold(fold, dcol_names):
                as_array = fold[dcol_names].to_numpy()
                return np.sum(np.sum(as_array, axis=0) ** 2).item()

            def create_balanced_splits(ss_df, dcol_names, k):
                ss_df = ss_df.reset_index(drop=True)
                ss_df = ss_df.sort_values('mod_psa', ascending=False)
                folds = []
                for n in range(ss_df.shape[0] + 1):
                    if n <= k:
                        if len(folds) == k:
                            continue
                        folds.append(ss_df.iloc[[n]])
                        continue
                    
                    last_row = ss_df.iloc[[n-1]]
                    scores = [
                        score_fold(pd.concat([fold, last_row], axis=0), dcol_names)# - score_fold(fold, dcol_names)
                        for fold in folds
                    ]

                    idx_of_best_fit = scores.index(min(scores))
                    folds[idx_of_best_fit] = pd.concat([folds[idx_of_best_fit], last_row], axis=0)

                fmap = None
                for i, fold in enumerate(folds):
                    fold['fold_id'] = i
                    if fmap is None:
                        fmap = fold
                    else:
                        fmap = pd.concat([fmap, fold], axis=0)

                return fmap

            dmap = mmap.copy()
            dmap['mod_psa'] = [np.floor(item) for item in dmap['psa'] / dmap['psa'].max() * 5]
            dmap = pd.get_dummies(dmap, columns=['age', 'isup'])
            dcol_names = [col for col in dmap.columns if col not in mmap.columns]
            dmap[dcol_names] = dmap[dcol_names].astype(int)
            
            jcol_name = 'case_id'
            dmap = dmap.groupby('case_id')[dcol_names].mean().reset_index()
            fmap = create_balanced_splits(
                ss_df=dmap[[jcol_name] + dcol_names],
                dcol_names=dcol_names,
                k=5,
            )

            keep_cols = [col for col in fmap.columns if not any(col.startswith(char) for char in ['mod_', 'age_', 'isup_'])]
            fmap = fmap[keep_cols]
            for _, row in fmap.iterrows():
                case_id = row['case_id']
                fold_id = row['fold_id']
                mmap.loc[mmap['case_id']==case_id, 'fold_id'] = fold_id

            fold_0 = mmap[mmap['fold_id'] < 4]
            fold_1 = mmap[~mmap['fold_id'].isin(fold_0['fold_id'])]
            fold_0 = fold_0.drop(columns=['fold_id'])
            fold_1 = fold_1.drop(columns=['fold_id'])
            #print(fold_0['psa'].std(), fold_1['psa'].std())
            #print(fold_0['isup'].std(), fold_1['isup'].std())
            return fold_0.sample(frac=1), fold_1.sample(frac=1)


    fold_splitter = BadMapGenerator(
        fpath_map_patchset=cfg.fpath_map_patchset,
        fpath_map_fold_0=cfg.fpath_map_fold_0,
        fpath_map_fold_1=cfg.fpath_map_fold_1,
    )


def testB():
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    def rm_h5(slide_id):
        return slide_id.rstrip('.h5')
    
    fpaths = cfg.dpath_qualityLog.iterdir()

    for fpath in tqdm(fpaths):
        qlog = pd.read_csv(fpath)
        #orids = qlog['slide_id']
        ori = qlog[[col for col in qlog.columns if col not in ['slide_id']]].to_numpy()
        
        qlog['slide_id'] = qlog['slide_id'].apply(rm_h5)
        #modids = qlog['slide_id']
        mod = qlog[[col for col in qlog.columns if col not in ['slide_id']]].to_numpy()

        #[print(o, m) for o, m in zip(orids, modids)]
        assert np.array_equal(ori, mod)
        qlog.to_csv(fpath, index=False)
        
def testC():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    def scatter_matrix(ss_df, dpath_diagnostics):
        sns.pairplot(ss_df)
        plt.savefig(dpath_diagnostics / 'quality_sm.png')

    #track_cols = ['on_bg', 'on_blur', 'on_dist']
    fpaths = cfg.dpath_qualityLog.iterdir()

    quality_stats = []
    for fpath in fpaths:
        qlog = pd.read_csv(fpath)
        quality_stats.append({
            'slide_id':fpath.name.rstrip('.csv'),
            'on_bg':qlog['on_bg'].mean().item(),
            'on_blur':qlog['on_blur'].mean().item(),
            'on_dist':qlog['on_dist'].mean().item(),
        })
        
    quality_stats = pd.DataFrame(quality_stats)
    print(quality_stats)
    scatter_matrix(quality_stats, cfg.dpath_diagnostics)
    
def testD():
    import pandas as pd
    def map_slide2patient(x, slide_id):
        pattern = slide_id.split('_')[0]
        return x.lstrip('patient_') == pattern
    
    slide_id = '001_ABC'
    patient_info = pd.read_csv(cfg.fpath_patientInfo)
    condition = patient_info['patient_n'].apply(map_slide2patient, args=(slide_id,))
    isup = patient_info.loc[condition, 'isup'].item()
    label = int(isup > 1)
    print(condition)
    print(isup)
    print(label)


def testE():
    print('xxx5.h5'.removesuffix('.h5'))
    print('xxx5.h5'.rstrip('.h5'))

def testF():
    import pandas as pd
    f0 = pd.read_csv(cfg.fpath_fold0)
    l0 = f0[f0['label'] == 0]
    l1 = f0[f0['label'] == 1]
    l0 = l0.sample(frac=1)
    l0 = l0.iloc[:l1.shape[0]]
    print(f0)
    print(f0['label'].mean())
    f0 = pd.concat([l0, l1], axis=0)
    f0 = f0.sample(frac=1)
    print(f0['label'].mean())
    print(f0)
    #f0.to_csv(cfg.fpath_fold0)


def testG():
    import torch
    from pathlib import Path
    dpath_pt = Path('data/1_featureRoot_bagsplit/ptFeature')
    fpaths = list(dpath_pt.iterdir())
    for fpath in fpaths:
        tensor = torch.load(fpath)
        print(tensor)
        print(tensor.shape)
        input('')


def testH():
    import numpy as np
    lst = [0.9, 0.68, 0.73, 0.42, 0.73]
    print(np.mean(sorted(lst)[len(lst) // 2:]))

def testI():
    import pandas as pd
    
    fpath_scores = Path('selection_scores_0.csv')
    ss_df = pd.read_csv(fpath_scores)

    #print(ss_df)
    case_id_freq = {}
    for _, row in ss_df.iterrows():
        unintended_string = row['rid']
        case_ids_of_run = unintended_string.replace(
            '[', ''
        ).replace(
            ']',''
        ).replace(
            "'",''
        ).replace(
            ' ',''
        ).split(',')

        for case_id in case_ids_of_run:
            if case_id in cfg.excl_by_id:
                continue
            if case_id in case_id_freq.keys():
                case_id_freq[case_id] += 1
            else:
                case_id_freq[case_id] = 1
    
    freqs = []
    for key in case_id_freq.keys():
        freqs.append({
            'slide_id':key,
            'freq':case_id_freq[key]
        })
    freqs = pd.DataFrame(freqs)

    encoding_map = pd.read_csv(cfg.fpath_encodingMap)
    encoding_map = encoding_map[encoding_map['slide_id'].isin(freqs['slide_id'])]
    freqs = freqs.merge(encoding_map, on='slide_id').sort_values(by='freq')
    print(encoding_map)
    print(freqs)
    #print(freqs)
    #import matplotlib.pyplot as plt
    #plt.hist(freqs['freq'], bins=50)
    #plt.show()


    

def testJ():
    import pandas as pd
    def get_slide_ids(string):
        return string.replace(
            '[', ''
        ).replace(
            ']',''
        ).replace(
            "'",''
        ).replace(
            ' ',''
        ).split(',')
    
    
    fpath_scores = Path('selection_scores_0.csv')
    ss_df = pd.read_csv(fpath_scores)

    #print(ss_df)
    freqs = []
    for _, row in ss_df.iterrows():
        slides_of_row = get_slide_ids(row['rid'])
        for slide_id in slides_of_row:
            if slide_id in cfg.excl_by_id:
                continue
            freqs.append({
                'slide_id':slide_id,
                'score':row['score'],
                'instance':1,
            })
    freqs = pd.DataFrame(freqs)
    
    ss_df = pd.read_csv(cfg.fpath_encodingMap)
    ss_df = ss_df[ss_df['slide_id'].isin(freqs['slide_id'].unique())]
    ss_df = ss_df[ss_df['label'] == 1]
    freqs = freqs[freqs['slide_id'].isin(ss_df['slide_id'].unique())]
    #print(freqs)


    freqs = freqs[freqs['score'] > 0.6].drop(columns='score')
    freqs = freqs.groupby(['slide_id']).sum().sort_values(by='instance')
    print(freqs)

    import matplotlib.pyplot as plt
    plt.hist(freqs['instance'], bins=50)
    plt.show()




def testK():
    import pandas as pd
    def get_slide_ids(string):
        return string.replace(
            '[', ''
        ).replace(
            ']',''
        ).replace(
            "'",''
        ).replace(
            ' ',''
        ).split(',')
    
    
    fpath_scores = Path('selection_scores_0.csv')
    ss_df = pd.read_csv(fpath_scores)

    #print(ss_df)
    freqs = []
    for _, row in ss_df.iterrows():
        slides_of_row = get_slide_ids(row['rid'])
        for slide_id in slides_of_row:
            if slide_id in cfg.excl_by_id_pos + cfg.excl_by_id_neg:
                continue
            freqs.append({
                'slide_id':slide_id,
                'score':row['score'],
                #'instance':1,
            })
    freqs = pd.DataFrame(freqs)
    
    ss_df = pd.read_csv(cfg.fpath_encodingMap)
    ss_df = ss_df[ss_df['slide_id'].isin(freqs['slide_id'].unique())]
    #ss_df = ss_df[ss_df['label'] == 1]
    freqs = freqs[freqs['slide_id'].isin(ss_df['slide_id'].unique())]
    #print(freqs)

    

    #freqs = freqs[freqs['score'] > 0.6].drop(columns='score')
    freqs = freqs.groupby(['slide_id']).mean().sort_values(by='score', ascending=False)
    labels = ss_df[['slide_id', 'label']]
    freqs = freqs.merge(labels, on='slide_id')
    print(freqs)
    freqs_0 = freqs[freqs['label']==0]
    freqs_1 = freqs[freqs['label']==1]

    import matplotlib.pyplot as plt
    plt.hist(freqs_0['score'], bins=50, alpha=0.3)
    plt.hist(freqs_1['score'], bins=50, alpha=0.3)
    plt.show()

def testL():
    import pandas as pd
    def get_slide_ids(string):
        return string.replace(
            '[', ''
        ).replace(
            ']',''
        ).replace(
            "'",''
        ).replace(
            ' ',''
        ).split(',')
    
    
    fpath_scores = Path('selection_scores_0.csv')
    omen_df = pd.read_csv(fpath_scores)
    #encd_df = pd.read_csv(cfg.fpath_encodingMap)
    #print(encd_df)
    #encd_df = encd_df[encd_df['slide_id'].isin(omen_df['slide_id'].unique())]
    #encd_df = encd_df[encd_df['label'] == 1]
    omen_df = omen_df.sort_values(by=['score'], ascending=False)
    omen_df = omen_df.reset_index(drop=True)
    print(omen_df)
    #omen_df = omen_df[omen_df['score']]

    for i, row in omen_df.iterrows():
        slide_ids = get_slide_ids(row['rid'])
        if i == 0:
            omens = slide_ids
        else:
            omens = [omen for omen in omens if omen in slide_ids]
            print(i, row['score'], omens)
        
    
def testM():
    """
    # ['003_GH', '037_DEF', '044_ABC', '048_G', '065_DEF', '073_ABC', '081_ABC', '091_DE', '099_ABC', '099_GHK', '144_AB', '151_AB', '205_ABC', '209_KL', '154_ABC', '085_CDE', '116_AB']
    x0 = ['001_ABC', '002_ABC', '002_DEF', '003_GH', '004_GHK', '007_DEF', '010_KL', '012_AB', '014_CD', '016_DEF', '016_GH', '016_KL', '017_DEF', '019_ABC', '020_AB', '026_DEF', '028_EF', '030_GH', '032_AB', '032_EF', '033_KL', '034_EF', '036_AB', '037_DEF', '038_DE', '039_DEF', '041_ABC', '042_DE', '044_ABC', '045_DEF', '048_G', '054_GH', '062_ABC', '065_DEF', '066_GH', '068_ABC', '068_DEF', '071_FGH', '072_ABC', '073_ABC', '074_ABC', '077_DEF', '078_CD', '080_ABC', '081_ABC', '082_ABC', '083_DEF', '091_AB', '091_DE', '097_ABC', '099_ABC', '099_GHK', '102_AB', '108_CD', '109_ABC', '110_DEF', '111_EF', '115_DEF', '118_ABC', '118_DEF', '119_L', '120_GH', '125_DEF', '129_DEF', '132_DEF', '135_ABC', '136_FG', '140_DEF', '141_EF', '144_AB', '144_CD', '144_EF', '145_LMN', '148_ABC', '148_DEF', '150_DEF', '151_AB', '152_DEF', '155_DEF', '158_CD', '160_A', '160_C', '167_CD', '169_DEF', '172_ABC', '172_KL', '173_FGH', '176_GHK', '177_DEF', '181_AB', '181_G', '183_DEF', '185_ABC', '189_EF', '198_EF', '205_ABC', '205_DEF', '208_DEF', '209_KL', '212_ABC', '123_EF', '151_CD', '171_M', '197_ABC', '201_KL', '124_AB', '154_ABC', '104_AB', '200_AB', '025_AB', '027_FG', '031_ABC', '049_DEF', '051_GH', '055_ABC', '074_DEF', '080_G', '085_CDE', '107_ABC', '107_L', '116_AB', '117_KL', '143_F', '146_DEF', '162_DEF', '163_ABC', '168_DEF', '193_ABCD', '203_ABC', '208_ABC', '210_L', '213_LMN', '164_DEF']
    x1 = ['001_ABC', '002_ABC', '003_GH', '005_DEF', '008_LMN', '009_ABC', '010_ABC', '012_CD', '012_E', '013_AB', '015_ZZZ', '016_ABC', '022_AB', '024_GH', '025_CD', '025_EF', '030_CD', '036_CD', '037_BC', '037_DEF', '038_ABC', '038_FGH', '044_ABC', '045_ABC', '045_DEF', '046_AB', '046_CD', '046_EF', '046_GH', '048_G', '051_DEF', '061_DEF', '064_DEF', '065_DEF', '066_GH', '068_DEF', '073_ABC', '073_DEF', '074_ABC', '076_ABD', '077_ABC', '081_ABC', '083_ABC', '087_DEF', '088_ABC', '091_DE', '092_ABC', '097_ABC', '099_ABC', '099_GHK', '100_DEF', '104_CD', '105_ABC', '108_CD', '111_AB', '119_GHK', '119_L', '120_GH', '122_DEF', '124_EF', '126_AB', '127_DEF', '128_DEF', '131_CD', '131_EF', '133_ABC', '133_GH', '134_ABC', '134_FGH', '141_EF', '142_ABC', '143_AB', '143_DE', '144_AB', '144_CD', '144_EF', '145_ABC', '149_DEF', '150_ABC', '151_AB', '159_ABC', '161_ABC', '165_AB', '169_ABC', '171_C', '171_L', '172_ABC', '172_KL', '176_ABC', '180_AB', '181_CD', '185_DEF', '187_ABC', '189_AB', '189_CD', '191_CD', '194_ABC', '195_AB', '198_AB', '205_ABC', '208_DEF', '209_KL', '123_EF', '187_DEF', '213_DEF', '031_F', '154_ABC', '013_EF', '048_ABC', '060_DEF', '072_DEF', '080_G', '085_CDE', '095_ABC', '107_DEF', '108_AB', '109_DEF', '116_AB', '119_DEF', '124_CD', '126_EF', '147_ABC', '150_GH', '153_ABC', '160_D', '164_ABC', '168_DEF', '193_ABCD', '206_CD', '210_DEF', '213_LMN', '113_DEF', '182_ABC']
    x2 = ['003_DEF', '003_GH', '006_ABC', '008_ABC', '010_DEF', '012_E', '013_AB', '014_AB', '016_KL', '017_ABC', '017_DEF', '018_DEF', '020_CD', '022_AB', '023_ABC', '025_CD', '026_ABC', '030_AB', '030_EF', '035_GH', '037_DEF', '038_DE', '042_DE', '044_ABC', '045_ABC', '046_GH', '047_EF', '048_DEF', '048_G', '051_DEF', '058_DEF', '059_CD', '059_EF', '064_ABC', '064_LMN', '065_ABC', '065_DEF', '066_DEF', '073_ABC', '081_ABC', '084_ABC', '084_DEF', '085_AB', '086_EF', '090_DEF', '091_DE', '094_ABC', '096_ABC', '097_ABC', '099_ABC', '099_GHK', '102_AB', '102_GH', '104_CD', '108_CD', '109_ABC', '111_CD', '111_EF', '126_AB', '127_DEF', '128_GH', '129_ABC', '131_CD', '132_DEF', '138_CD', '140_KL', '141_EF', '144_AB', '148_DEF', '151_AB', '152_DEF', '158_AB', '158_CD', '159_ABC', '163_DEF', '171_E', '172_ABC', '173_ABC', '175_DEF', '176_GHK', '179_ABC', '181_AB', '181_CD', '184_ZZZ', '185_DEF', '186_GH', '189_CD', '191_CD', '194_ABC', '196_DE', '205_ABC', '209_KL', '213_ABC', '167_AB', '187_DEF', '031_F', '154_ABC', '013_EF', '028_CD', '031_ABC', '035_ABC', '050_ABC', '054_DEF', '071_KL', '072_DEF', '080_G', '085_CDE', '085_F', '087_ABC', '090_ABC', '097_DEF', '102_CD', '107_L', '108_AB', '109_DEF', '116_AB', '119_DEF', '124_CD', '137_GHK', '138_AB', '139_GH', '147_ABC', '153_ABC', '166_DEF', '172_DEF', '202_ABC', '203_ABC', '205_GH', '206_EF', '210_ABC', '210_GHK', '182_ABC', '191_EFG']
    x3 = ['001_ABC', '003_DEF', '003_GH', '004_ABC', '005_ABC', '008_LMN', '010_DEF', '010_GH', '010_KL', '012_AB', '012_CD', '014_AB', '017_DEF', '020_CD', '022_AB', '033_GH', '033_KL', '034_CD', '034_EF', '037_DEF', '038_FGH', '041_ABC', '043_AB', '044_ABC', '046_EF', '047_EF', '048_G', '049_ABC', '059_EF', '061_ABC', '065_ABC', '065_DEF', '066_DEF', '067_AB', '073_ABC', '077_DEF', '081_ABC', '082_ABC', '085_AB', '086_EF', '087_DEF', '091_AB', '091_DE', '097_ABC', '099_ABC', '099_GHK', '100_DEF', '103_DEF', '104_CD', '108_CD', '109_ABC', '111_CD', '112_ABC', '120_ABC', '120_GH', '125_ABC', '127_DEF', '130_DEF', '133_ABC', '134_FGH', '135_DEF', '141_EF', '142_DEF', '144_AB', '145_GHK', '146_ABC', '147_DEF', '148_DEF', '150_ABC', '151_AB', '152_DEF', '158_CD', '159_ABC', '165_CDE', '171_B', '171_C', '171_D', '171_G', '171_L', '173_ABC', '174_ABC', '175_ABC', '178_ABC', '179_DEF', '180_AB', '183_DEF', '186_ABC', '186_GH', '187_ABC', '188_ABC', '189_CD', '194_ABC', '204_ABC', '205_ABC', '209_KL', '211_ZZZ', '213_ABC', '093_CD', '139_KL', '174_DEF', '187_DEF', '197_ABC', '154_ABC', '027_DE', '028_AB', '031_ABC', '050_ABC', '053_GH', '055_ABC', '057_DEF', '069_CD', '070_GHK', '071_DE', '085_CDE', '107_L', '109_DEF', '116_AB', '133_DEF', '137_GHK', '143_C', '172_DEF', '179_GH', '180_CD', '193_EFG', '202_ABC', '203_ABC', '203_GHK', '206_CD', '208_ABC', '210_DEF', '213_LMN', '182_ABC', '191_EFG']
    x4 = ['001_ABC', '002_DEF', '003_GH', '004_DEF', '004_LMN', '005_ABC', '008_ABC', '008_GHK', '010_KL', '016_ABC', '016_DEF', '016_GH', '016_KL', '017_ABC', '019_DEF', '025_CD', '030_AB', '033_ABC', '033_GH', '034_EF', '036_AB', '036_CD', '037_BC', '037_DEF', '038_DE', '043_CD', '044_ABC', '045_DEF', '046_CD', '046_EF', '047_AB', '048_DEF', '048_G', '059_CD', '065_DEF', '066_GH', '068_ABC', '073_ABC', '073_DEF', '076_EF', '077_KL', '081_ABC', '085_AB', '089_EF', '091_AB', '091_DE', '096_DEF', '097_ABC', '098_ABC', '099_ABC', '099_GHK', '101_ABC', '101_DEF', '102_AB', '102_GH', '103_ABC', '104_CD', '104_EF', '110_ABC', '113_ABC', '114_AB', '118_DEF', '120_ABC', '121_CD', '123_CD', '125_ABC', '131_EF', '140_DEF', '141_AB', '141_CD', '142_ABC', '143_DE', '144_AB', '145_LMN', '151_AB', '151_EF', '155_DEF', '158_CD', '165_CDE', '167_CD', '171_B', '171_D', '171_H', '171_K', '171_L', '172_KL', '178_ABC', '179_ABC', '179_DEF', '181_G', '186_ABC', '189_EF', '190_DEF', '191_CD', '198_AB', '205_ABC', '208_DEF', '209_KL', '212_DEF', '098_GHK', '031_F', '120_DEF', '154_ABC', '168_ABC', '180_EFG', '104_AB', '027_DE', '040_DEF', '048_ABC', '049_DEF', '050_DEF', '051_GH', '060_DEF', '062_DEF', '072_DEF', '085_CDE', '095_ABC', '105_DEF', '116_AB', '117_DE', '126_CD', '126_EF', '134_KL', '139_GH', '147_ABC', '163_ABC', '166_DEF', '176_DEF', '198_CD', '202_ABC', '203_ABC', '206_CD', '207_FG']
    x5 = ['002_ABC', '003_GH', '005_KL', '007_DEF', '009_GH', '010_DEF', '016_DEF', '017_ABC', '017_DEF', '019_ABC', '020_CD', '021_ABC', '021_DEF', '024_KL', '026_ABC', '030_CD', '031_DE', '033_DEF', '033_KL', '034_AB', '034_CD', '036_CD', '037_A', '037_BC', '037_DEF', '038_ABC', '041_DEF', '042_DE', '044_ABC', '045_ABC', '048_G', '054_KL', '058_DEF', '059_CD', '061_DEF', '065_DEF', '066_GH', '068_ABC', '071_FGH', '073_ABC', '077_KL', '081_ABC', '082_DEF', '085_AB', '086_AB', '086_EF', '087_DEF', '088_DEF', '088_KL', '090_DEF', '091_DE', '092_ABC', '096_DEF', '099_ABC', '099_GHK', '100_DEF', '101_ABC', '101_DEF', '102_AB', '103_ABC', '104_CD', '105_ABC', '108_CD', '114_AB', '114_EF', '115_ABC', '115_DEF', '120_ABC', '123_CD', '125_ABC', '129_ABC', '131_EF', '134_DE', '138_CD', '140_ABC', '144_AB', '145_ABC', '145_GHK', '147_DEF', '151_AB', '152_DEF', '161_ABC', '167_CD', '167_EF', '171_D', '179_ABC', '180_AB', '190_ABC', '190_DEF', '196_DE', '199_DEF', '205_ABC', '205_DEF', '209_GH', '209_KL', '211_ZZZ', '093_CD', '102_EF', '132_ABC', '171_M', '031_F', '102_MN', '154_ABC', '168_ABC', '200_AB', '040_DEF', '055_ABC', '060_DEF', '070_ABC', '070_DEF', '070_GHK', '071_ABC', '071_KL', '074_GH', '085_CDE', '105_DEF', '107_ABC', '107_DEF', '108_AB', '109_DEF', '116_AB', '119_DEF', '126_CD', '138_AB', '147_ABC', '153_ABC', '164_ABC', '180_CD', '196_ABC', '208_ABC', '210_GHK', '213_LMN', '191_EFG']
    x6 = ['002_G', '003_GH', '006_ABC', '006_DEF', '009_DEF', '009_GH', '010_KL', '012_E', '014_EF', '016_DEF', '019_ABC', '019_DEF', '022_AB', '023_ABC', '024_KL', '025_CD', '028_EF', '032_CD', '033_ABC', '033_GH', '036_AB', '036_CD', '037_BC', '037_DEF', '038_DE', '044_ABC', '044_DEF', '047_CD', '048_DEF', '048_G', '051_DEF', '065_DEF', '068_DEF', '070_LMN', '072_ABC', '073_ABC', '077_KL', '081_ABC', '084_DEF', '086_CD', '089_EF', '091_DE', '092_DEF', '094_ABC', '096_ABC', '097_ABC', '099_ABC', '099_GHK', '102_AB', '104_CD', '109_ABC', '110_ABC', '110_DEF', '111_EF', '113_ABC', '125_ABC', '127_ABC', '127_DEF', '129_ABC', '134_ABC', '134_FGH', '136_FG', '138_CD', '141_CD', '144_AB', '145_ABC', '147_DEF', '148_ABC', '149_DEF', '151_AB', '155_DEF', '159_DEF', '160_K', '161_ABC', '167_CD', '167_EF', '169_ABC', '169_DEF', '174_ABC', '177_DEF', '181_CD', '181_G', '188_ABC', '190_ABC', '191_CD', '195_CD', '204_DEF', '205_ABC', '209_KL', '212_ABC', '213_ABC', '167_AB', '209_DEF', '213_DEF', '031_F', '154_ABC', '166_ABC', '200_AB', '013_CD', '040_DEF', '055_ABC', '062_DEF', '069_F', '070_GHK', '071_ABC', '071_DE', '072_DEF', '074_GH', '080_G', '085_CDE', '085_F', '087_ABC', '095_ABC', '098_DEF', '100_ABC', '102_KL', '107_GHK', '116_AB', '130_ABC', '138_AB', '150_GH', '153_ABC', '155_ABC', '164_ABC', '168_DEF', '179_GH', '193_EFG', '201_ABC', '206_EF', '210_GHK', '164_DEF', '182_ABC', '191_AB']
    x7 = ['001_DEF', '003_GH', '005_KL', '006_ABC', '006_DEF', '007_ABC', '007_DEF', '009_ABC', '009_DEF', '013_AB', '014_AB', '020_AB', '021_DEF', '025_CD', '026_DEF', '028_EF', '029_ABC', '034_EF', '037_DEF', '044_ABC', '045_DEF', '047_EF', '048_DEF', '048_G', '049_GHK', '049_L', '059_AB', '061_ABC', '062_ABC', '063_ABC', '064_ABC', '065_DEF', '068_ABC', '073_ABC', '081_ABC', '084_ABC', '084_DEF', '091_DE', '092_DEF', '099_ABC', '099_GHK', '100_DEF', '102_AB', '104_EF', '108_CD', '110_ABC', '111_EF', '119_GHK', '120_ABC', '123_CD', '124_EF', '125_ABC', '127_DEF', '129_ABC', '132_DEF', '133_GH', '134_FGH', '141_CD', '142_DEF', '143_AB', '144_AB', '146_ABC', '148_ABC', '151_AB', '152_DEF', '159_ABC', '160_C', '163_DEF', '165_AB', '167_CD', '167_EF', '171_L', '172_KL', '174_ABC', '175_ABC', '178_DE', '180_AB', '186_ABC', '186_GH', '187_ABC', '189_EF', '192_DEF', '194_DEF', '195_CD', '199_ABC', '204_ABC', '205_ABC', '205_DEF', '209_KL', '093_CD', '123_EF', '151_CD', '171_M', '174_DEF', '197_ABC', '213_DEF', '213_GHK', '114_CD', '089_CD', '120_DEF', '154_ABC', '168_ABC', '183_ABC', '136_DE', '013_CD', '024_ABC', '028_CD', '049_DEF', '050_ABC', '053_GH', '054_DEF', '070_DEF', '080_G', '085_CDE', '085_F', '095_ABC', '095_DEF', '105_DEF', '105_GHK', '107_GHK', '116_AB', '124_CD', '137_GHK', '139_GH', '153_DEF', '169_GH', '180_CD', '188_DEF', '203_GHK', '210_GHK', '210_L', '213_LMN', '191_EFG']
    """

    x0 = ['003_ABC', '003_GH', '004_ABC', '005_ABC', '005_DEF', '005_KL', '006_ABC', '006_DEF', '007_ABC', '011_ABC', '016_GH', '017_DEF', '022_CD', '022_EF', '026_DEF', '027_HKL', '031_DE', '033_DEF', '034_AB', '035_DEF', '036_EF', '037_A', '037_DEF', '038_ABC', '038_FGH', '042_ABC', '044_ABC', '047_CD', '048_G', '065_DEF', '066_DEF', '073_ABC', '073_DEF', '077_DEF', '078_CD', '081_ABC', '082_DEF', '084_DEF', '085_AB', '086_AB', '086_CD', '087_DEF', '088_DEF', '091_AB', '091_DE', '094_DEF', '096_DEF', '098_ABC', '099_ABC', '099_GHK', '102_GH', '103_ABC', '109_ABC', '110_ABC', '110_DEF', '111_CD', '114_AB', '120_GH', '124_EF', '125_DEF', '126_AB', '129_DEF', '131_AB', '133_ABC', '133_GH', '140_GH', '144_AB', '149_DEF', '151_AB', '152_DEF', '155_DEF', '159_ABC', '160_L', '161_ABC', '163_DEF', '165_AB', '169_ABC', '170_GH', '171_C', '171_F', '171_G', '183_DEF', '189_CD', '189_EF', '194_ABC', '195_AB', '196_DE', '198_AB', '199_ABC', '204_DEF', '205_ABC', '205_KL', '209_KL', '080_DEF', '093_CD', '123_EF', '171_M', '192_ABC', '197_ABC', '213_GHK', '005_GH', '089_CD', '154_ABC', '168_ABC', '104_AB', '028_AB', '053_GH', '054_DEF', '060_ABC', '060_DEF', '072_DEF', '074_GH', '085_CDE', '097_DEF', '108_AB', '109_DEF', '116_AB', '126_CD', '137_ABC', '138_AB', '146_DEF', '162_DEF', '163_ABC', '164_ABC', '168_DEF', '179_GH', '193_ABCD', '193_EFG', '202_ABC', '203_LMN', '210_ABC', '191_AB', '191_EFG']
    x1 = ['001_ABC', '002_ABC', '003_GH', '004_LMN', '005_KL', '010_GH', '012_CD', '015_ZZZ', '016_ABC', '016_GH', '017_DEF', '019_DEF', '020_CD', '022_AB', '023_ABC', '023_DEF', '026_ABC', '027_HKL', '030_EF', '030_GH', '032_GH', '033_KL', '034_AB', '036_EF', '037_DEF', '038_DE', '042_ABC', '043_CD', '044_ABC', '045_ABC', '046_AB', '046_EF', '048_DEF', '048_G', '049_GHK', '049_L', '051_ABC', '054_GH', '058_GH', '061_ABC', '065_DEF', '066_DEF', '073_ABC', '077_GH', '081_ABC', '084_ABC', '088_DEF', '091_AB', '091_DE', '096_ABC', '098_ABC', '099_ABC', '099_DEF', '099_GHK', '100_DEF', '102_GH', '103_DEF', '104_CD', '115_ABC', '118_ABC', '118_DEF', '119_L', '120_ABC', '122_DEF', '127_DEF', '131_AB', '133_ABC', '134_DE', '134_FGH', '140_ABC', '141_CD', '144_AB', '147_DEF', '148_ABC', '149_DEF', '151_AB', '160_A', '165_CDE', '167_EF', '171_N', '172_KL', '173_ABC', '177_ABC', '177_DEF', '179_ABC', '182_DEF', '185_ABC', '194_DEF', '197_DEF', '203_DEF', '205_ABC', '208_DEF', '209_KL', '212_DEF', '213_ABC', '089_AB', '102_EF', '213_DEF', '160_E', '005_GH', '124_AB', '154_ABC', '168_ABC', '183_ABC', '136_DE', '200_AB', '013_EF', '025_AB', '048_ABC', '051_GH', '055_DEF', '072_DEF', '074_DEF', '085_CDE', '095_DEF', '100_ABC', '102_CD', '108_AB', '109_DEF', '116_AB', '117_FGH', '137_GHK', '146_DEF', '150_KL', '153_DEF', '195_EF', '201_DEF', '202_ABC', '203_ABC', '207_DE', '210_L', '213_LMN', '164_DEF']
    x2 = ['002_DEF', '003_DEF', '003_GH', '010_DEF', '010_KL', '012_E', '014_AB', '014_EF', '017_ABC', '019_ABC', '022_AB', '024_DEF', '029_DEF', '030_CD', '033_GH', '034_EF', '037_BC', '037_DEF', '038_DE', '041_DEF', '044_ABC', '045_DEF', '046_EF', '048_G', '049_ABC', '051_DEF', '061_ABC', '061_DEF', '065_DEF', '066_DEF', '066_GH', '068_ABC', '073_ABC', '073_DEF', '075_DEF', '077_GH', '081_ABC', '082_DEF', '088_GH', '090_DEF', '091_DE', '092_ABC', '094_DEF', '099_ABC', '099_DEF', '099_GHK', '101_DEF', '102_GH', '104_CD', '104_EF', '115_DEF', '122_DEF', '130_DEF', '132_DEF', '134_ABC', '134_DE', '134_FGH', '136_FG', '140_DEF', '140_KL', '141_AB', '142_ABC', '144_AB', '144_EF', '149_DEF', '150_ABC', '151_AB', '151_EF', '160_B', '165_CDE', '170_DEF', '170_GH', '171_A', '171_D', '171_F', '172_ABC', '178_ABC', '180_AB', '181_AB', '181_EF', '189_CD', '190_DEF', '194_DEF', '195_AB', '198_EF', '199_DEF', '205_ABC', '205_DEF', '209_KL', '211_ZZZ', '212_ABC', '080_DEF', '093_CD', '151_CD', '187_DEF', '213_GHK', '114_CD', '089_CD', '154_ABC', '160_P', '180_EFG', '183_ABC', '025_AB', '028_CD', '055_ABC', '058_ABC', '060_ABC', '060_DEF', '069_F', '070_ABC', '072_DEF', '074_GH', '085_CDE', '085_F', '087_ABC', '095_DEF', '106_ABC', '108_AB', '109_DEF', '116_AB', '130_ABC', '143_F', '147_ABC', '162_ABC', '169_GH', '180_CD', '193_EFG', '203_LMN', '207_DE', '210_ABC', '210_GHK', '164_DEF', '182_ABC']
    x3 = ['003_DEF', '003_GH', '006_ABC', '008_ABC', '010_DEF', '012_E', '013_AB', '014_AB', '016_KL', '017_ABC', '017_DEF', '018_DEF', '020_CD', '022_AB', '023_ABC', '025_CD', '026_ABC', '030_AB', '030_EF', '035_GH', '037_DEF', '038_DE', '042_DE', '044_ABC', '045_ABC', '046_GH', '047_EF', '048_DEF', '048_G', '051_DEF', '058_DEF', '059_CD', '059_EF', '064_ABC', '064_LMN', '065_ABC', '065_DEF', '066_DEF', '073_ABC', '081_ABC', '084_ABC', '084_DEF', '085_AB', '086_EF', '090_DEF', '091_DE', '094_ABC', '096_ABC', '097_ABC', '099_ABC', '099_GHK', '102_AB', '102_GH', '104_CD', '108_CD', '109_ABC', '111_CD', '111_EF', '126_AB', '127_DEF', '128_GH', '129_ABC', '131_CD', '132_DEF', '138_CD', '140_KL', '141_EF', '144_AB', '148_DEF', '151_AB', '152_DEF', '158_AB', '158_CD', '159_ABC', '163_DEF', '171_E', '172_ABC', '173_ABC', '175_DEF', '176_GHK', '179_ABC', '181_AB', '181_CD', '184_ZZZ', '185_DEF', '186_GH', '189_CD', '191_CD', '194_ABC', '196_DE', '205_ABC', '209_KL', '213_ABC', '167_AB', '187_DEF', '031_F', '154_ABC', '013_EF', '028_CD', '031_ABC', '035_ABC', '050_ABC', '054_DEF', '071_KL', '072_DEF', '080_G', '085_CDE', '085_F', '087_ABC', '090_ABC', '097_DEF', '102_CD', '107_L', '108_AB', '109_DEF', '116_AB', '119_DEF', '124_CD', '137_GHK', '138_AB', '139_GH', '147_ABC', '153_ABC', '166_DEF', '172_DEF', '202_ABC', '203_ABC', '205_GH', '206_EF', '210_ABC', '210_GHK', '182_ABC', '191_EFG']
    x4 = ['003_GH', '005_DEF', '005_KL', '006_DEF', '009_DEF', '009_GH', '010_KL', '012_AB', '012_E', '019_DEF', '022_EF', '026_DEF', '030_CD', '030_EF', '030_GH', '031_DE', '033_GH', '033_KL', '034_CD', '035_GH', '036_EF', '037_DEF', '038_DE', '039_ABC', '042_ABC', '044_ABC', '047_AB', '048_DEF', '048_G', '049_GHK', '049_L', '052_ABC', '061_ABC', '061_DEF', '065_DEF', '066_DEF', '068_ABC', '070_LMN', '073_ABC', '073_DEF', '077_DEF', '077_GH', '077_KL', '081_ABC', '082_ABC', '087_DEF', '091_DE', '092_ABC', '098_LMN', '099_ABC', '099_GHK', '100_DEF', '102_AB', '103_ABC', '108_CD', '110_ABC', '110_DEF', '111_AB', '111_CD', '111_EF', '114_EF', '123_CD', '134_FGH', '144_AB', '149_DEF', '151_AB', '151_EF', '152_DEF', '159_DEF', '160_L', '160_R', '161_ABC', '170_DEF', '170_GH', '171_B', '171_F', '175_ABC', '178_ABC', '178_DE', '178_FGH', '180_AB', '181_CD', '182_DEF', '183_DEF', '186_GH', '189_AB', '189_CD', '194_ABC', '199_ABC', '199_DEF', '203_DEF', '205_ABC', '205_KL', '209_KL', '212_ABC', '212_DEF', '089_AB', '093_CD', '123_EF', '132_ABC', '151_CD', '167_AB', '209_DEF', '031_F', '120_DEF', '154_ABC', '168_ABC', '180_EFG', '183_ABC', '031_ABC', '048_ABC', '050_DEF', '051_GH', '054_ABC', '057_DEF', '070_GHK', '072_DEF', '080_G', '085_CDE', '106_ABC', '107_ABC', '116_AB', '117_FGH', '124_CD', '126_CD', '130_ABC', '137_DEF', '143_F', '169_GH', '180_CD', '188_DEF', '202_DEF', '210_ABC']
    x5 = ['001_DEF', '003_GH', '004_GHK', '005_DEF', '006_ABC', '009_ABC', '011_DEF', '012_AB', '014_CD', '019_ABC', '020_AB', '022_CD', '023_DEF', '033_ABC', '033_KL', '034_EF', '036_EF', '037_BC', '037_DEF', '040_ABC', '043_AB', '044_ABC', '047_CD', '048_G', '054_GH', '054_KL', '061_ABC', '065_DEF', '073_ABC', '074_ABC', '076_ABD', '077_DEF', '077_GH', '081_ABC', '082_DEF', '083_ABC', '084_ABC', '088_ABC', '090_GH', '091_AB', '091_DE', '099_ABC', '099_GHK', '101_DEF', '102_GH', '103_DEF', '108_CD', '113_ABC', '114_AB', '115_ABC', '120_ABC', '125_DEF', '127_DEF', '129_ABC', '134_FGH', '135_ABC', '136_FG', '138_CD', '141_CD', '144_AB', '145_ABC', '145_DEF', '148_ABC', '149_DEF', '151_AB', '159_ABC', '160_F', '165_CDE', '167_CD', '170_ABC', '171_A', '171_B', '171_C', '171_E', '171_H', '171_K', '171_L', '178_ABC', '179_ABC', '182_DEF', '189_AB', '189_CD', '189_EF', '190_ABC', '190_DEF', '194_DEF', '197_DEF', '198_AB', '198_EF', '199_DEF', '205_ABC', '209_GH', '209_KL', '119_ABC', '151_CD', '156_ABC', '213_DEF', '005_GH', '089_CD', '154_ABC', '178_KL', '027_DE', '028_AB', '028_CD', '035_ABC', '048_ABC', '051_GH', '054_ABC', '057_DEF', '060_DEF', '071_DE', '074_DEF', '080_G', '085_CDE', '087_ABC', '100_ABC', '107_DEF', '109_DEF', '116_AB', '126_CD', '138_AB', '147_ABC', '163_ABC', '169_GH', '172_DEF', '202_DEF', '206_CD', '206_EF', '207_DE', '208_ABC', '078_EF', '191_AB', '191_EFG']
    x6 = ['001_DEF', '002_ABC', '003_GH', '004_GHK', '006_ABC', '007_ABC', '010_DEF', '013_AB', '017_DEF', '021_ABC', '022_EF', '024_DEF', '025_CD', '033_DEF', '033_GH', '034_AB', '036_CD', '037_DEF', '038_FGH', '041_DEF', '043_CD', '044_ABC', '047_EF', '048_DEF', '048_G', '051_DEF', '061_DEF', '064_DEF', '065_DEF', '068_DEF', '072_ABC', '073_ABC', '076_ABD', '077_KL', '081_ABC', '083_DEF', '084_DEF', '085_AB', '086_CD', '088_ABC', '088_DEF', '091_AB', '091_DE', '092_DEF', '094_ABC', '099_ABC', '099_GHK', '104_CD', '109_ABC', '111_CD', '118_ABC', '119_GHK', '124_EF', '125_ABC', '125_DEF', '126_AB', '127_ABC', '129_ABC', '131_EF', '136_FG', '140_DEF', '140_KL', '141_AB', '142_DEF', '143_DE', '144_AB', '151_AB', '152_ABC', '152_DEF', '158_AB', '158_CD', '159_ABC', '160_F', '160_L', '160_R', '161_ABC', '165_AB', '165_CDE', '167_CD', '167_EF', '170_DEF', '170_GH', '171_A', '171_E', '171_L', '171_N', '172_ABC', '175_ABC', '175_DEF', '179_DEF', '189_AB', '189_CD', '195_AB', '196_DE', '197_DEF', '205_ABC', '205_DEF', '209_KL', '212_ABC', '213_ABC', '089_AB', '102_EF', '132_ABC', '139_KL', '167_AB', '174_DEF', '031_F', '102_MN', '154_ABC', '104_AB', '035_ABC', '048_ABC', '050_ABC', '051_GH', '072_DEF', '085_CDE', '085_F', '087_ABC', '097_DEF', '102_CD', '107_ABC', '116_AB', '130_ABC', '134_KL', '143_C', '147_ABC', '153_ABC', '160_D', '163_ABC', '168_DEF', '169_GH', '210_GHK', '113_DEF']
    x7 = ['001_ABC', '003_DEF', '003_GH', '005_ABC', '005_DEF', '005_KL', '006_ABC', '010_KL', '012_E', '014_CD', '014_EF', '016_GH', '017_DEF', '018_DEF', '019_DEF', '025_CD', '027_ABC', '034_CD', '036_AB', '036_EF', '037_DEF', '044_ABC', '045_ABC', '046_CD', '047_AB', '048_G', '053_ABC', '058_DEF', '059_CD', '061_DEF', '065_ABC', '065_DEF', '073_ABC', '076_EF', '077_DEF', '077_KL', '078_CD', '081_ABC', '082_ABC', '087_DEF', '089_EF', '091_AB', '091_DE', '099_ABC', '099_GHK', '100_DEF', '102_AB', '103_DEF', '104_CD', '104_EF', '105_ABC', '110_ABC', '110_DEF', '111_CD', '111_EF', '115_ABC', '118_ABC', '120_ABC', '124_EF', '126_AB', '130_DEF', '131_EF', '134_DE', '136_FG', '140_ABC', '140_GH', '144_AB', '151_AB', '152_ABC', '158_AB', '159_ABC', '163_DEF', '167_EF', '169_ABC', '169_DEF', '170_GH', '171_C', '171_L', '181_EF', '181_G', '182_DEF', '184_ZZZ', '185_ABC', '189_AB', '198_AB', '198_EF', '199_DEF', '204_DEF', '205_ABC', '209_ABC', '209_KL', '080_DEF', '102_EF', '161_DEF', '170_KL', '197_ABC', '213_DEF', '213_GHK', '114_CD', '005_GH', '063_DEF', '102_MN', '120_DEF', '124_AB', '154_ABC', '104_AB', '024_ABC', '025_AB', '028_CD', '048_ABC', '049_DEF', '054_ABC', '055_DEF', '060_ABC', '070_ABC', '085_CDE', '085_F', '102_CD', '107_L', '116_AB', '139_ABC', '146_DEF', '150_KL', '163_ABC', '172_DEF', '203_ABC', '203_GHK', '205_GH', '207_ABC', '207_FG', '210_L', '191_AB', '191_EFG']

    xx = [x for x in x0 if x in x1]
    print(len(xx)/len(x0))
    xx = [x for x in xx if x in x2]
    print(len(xx)/len(x0))
    xx = [x for x in xx if x in x3]
    print(len(xx)/len(x0))
    xx = [x for x in xx if x in x4]
    print(len(xx)/len(x0))
    xx = [x for x in xx if x in x5]
    print(len(xx)/len(x0))
    xx = [x for x in xx if x in x6]
    print(len(xx)/len(x0))
    xx = [x for x in xx if x in x7]
    print(len(xx)/len(x0))
    print(xx)




def testN():
    import numpy as np
    vec = np.array([0.0, 0.2857142853061225, 0.3636363635261708, 0.2758620688703924, 0.2222222219753086, 0.3333333332323232, 0.2857142856122449, 0.0, 0.0, 0.2857142856559767, 0.2631578946906741, 0.0, 0.23076923068047336, 0.32432432423666907, 0.23809523798185941, 0.0, 0.0, 0.22222222213991769, 0.22222222213991769, 0.13333333324444444, 0.20833333324652778, 0.21739130425330813, 0.21739130425330813, 0.249999999875, 0.2758620688703924, 0.31249999990234373, 0.352941176366782, 0.0, 0.21052631567867036, 0.3235294116695502])
    vec = [[0, item][int(item > 0.5)] for item in vec]
    
    score = 0
    for i in range(3, len(vec)+1):
        window = vec[i-3:i]
        score += np.prod(window) / len(vec)
    print(score)

def testO():
    import pandas as pd
    from tqdm import tqdm
    ss_df = pd.read_csv('ss_fold_specified.csv')
    birds = []
    for _, row in ss_df.iterrows():
        bird_line = get_slide_ids(row['incl_of_fold0'])
        [birds.append(bird) for bird in bird_line if not bird in birds]

    weights = {}
    for bird in birds:
        weights[bird] = 1
    weights['error_rate'] = -1
    
    lr, breaking_point = 0.005, 10

    N = 500
    record = []
    for i in tqdm(range(N)):
        error_rate = 0
        df = ss_df.sample(frac=0.5)
        for _, row in df.iterrows():
            bird_line = get_slide_ids(row['incl_of_fold0'])
            load = 0
            for bird in bird_line:
                load += weights[bird]
            linebreak = load >= breaking_point

            if linebreak and row['score'] == -1:
                error_rate += 1 / df.shape[0]
                for bird in bird_line:
                    weights[bird] -= lr * len(bird_line) # final factor probably only helps because the small birdlines are less reliable as datapoints (for some reason)
                    #weights[bird] = max([weights[bird], 0]) # or maybe i have too many datapoits (though should be better anyway)
            elif not linebreak and row['score'] == 0:
                error_rate += 1 / df.shape[0]
                for bird in bird_line:
                    weights[bird] += lr * len(bird_line)
        weights['error_rate'] = error_rate
        record.append(weights.copy())

    record = pd.DataFrame(record)
    #print(record)

    highlight = record.iloc[-1].nlargest(12).index.tolist()
    lowlight = record.iloc[-1].nsmallest(12).index.tolist()
    print(f"high: {highlight}")
    print(f"low: {lowlight}")
    #import matplotlib.pyplot as plt
    #plt.plot(record[highlight])
    #plt.plot(record['error_rate'])
    #plt.plot(record[record.columns[record.iloc[-1].argmax()]])
    #plt.plot(record[record.columns[record.iloc[-1].argmin()]])
    #plt.ylim([0, max(record['error_rate'])])
    #plt.show()
    

    #['031_ABC', '176_GHK', '201_KL', '144_EF', '210_L', '110_DEF', '148_ABC'|'039_DEF', '200_AB', '071_FGH', '162_DEF', '189_EF', '001_ABC', '091_AB']
    #['176_GHK', '031_ABC', '144_EF', '177_DEF', '072_ABC', '163_ABC', '158_CD'|'042_DE', '007_DEF', '026_DEF', '118_DEF', '107_L', '140_DEF', '091_AB']
    #

    #high: ['140_DEF', '001_ABC', '125_DEF', '042_DE', '091_AB']
    #low: ['176_GHK', '144_EF', '144_CD', '028_EF', '115_DEF']
    #high: ['091_AB', '189_EF', '071_FGH', '104_AB', '125_DEF']
    #low: ['144_EF', '031_ABC', '210_L', '144_CD', '176_GHK']

def testP():
    """
    import pandas as pd
    df = pd.read_csv('ss_most_data0.csv')
    df['label0_ratio'] = -1
    df = df[['rid', 'score', 'label0_ratio', 'slides_of_fold0']]
    print(df)
    df.to_csv('ss_most_data.csv', index=False)
    """

def testQ():
    pass

def testR():
    pass

def testS():
    pass

def testT():
    pass

testQ()
testR()
testS()
testT()



raise SystemExit
testA()
testB()
testC()
testD()
testE()
testF()
testG()
testH()
testI()
testJ()
testK()
testL()
testM()
testN()
testO()
testP()