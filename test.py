
from config import *

def test0():
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

            def create_balanced_splits(df, dcol_names, k):
                df = df.reset_index(drop=True)
                df = df.sort_values('mod_psa', ascending=False)
                folds = []
                for n in range(df.shape[0] + 1):
                    if n <= k:
                        if len(folds) == k:
                            continue
                        folds.append(df.iloc[[n]])
                        continue
                    
                    last_row = df.iloc[[n-1]]
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
                df=dmap[[jcol_name] + dcol_names],
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


def test1():
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
        
def test2():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    def scatter_matrix(df, dpath_diagnostics):
        sns.pairplot(df)
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
    
def test3():
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


def test4():
    print('xxx5.h5'.removesuffix('.h5'))
    print('xxx5.h5'.rstrip('.h5'))
    



        

    


#test0()
#test1()
#test2()
#test3()
test4()



