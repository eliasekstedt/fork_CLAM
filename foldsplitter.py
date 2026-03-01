import re
import numpy as np
import pandas as pd

class FeatureMapSplitter:
    def __init__(self, fpath_encodingMap, excl_by_id, fpath_fold0, fpath_fold1, rid):
        mmap = pd.read_csv(fpath_encodingMap)
        all_ids = mmap['slide_id'].to_list()
        mmap = mmap[~mmap['slide_id'].isin(excl_by_id)]

        pex = mmap[mmap['label'] == 1].sample(frac=0.9)
        nex = mmap[mmap['label'] == 0].sample(frac=0.75)
        mmap = pd.concat([pex, nex], axis=0).sample(frac=1)
        remaining_ids = mmap['slide_id'].to_list()
        fltrd_ids = [id for id in all_ids if id not in remaining_ids]
        #print(mmap)
        message = f"filtered_ids:\n{fltrd_ids}"
        with open('fltrlog.txt', 'a') as file:
            file.write(f'{rid}\n{message}\n')
        #raise SystemExit

        mmap['case_id'] = mmap['slide_id'].apply(self.get_case_id)
        fold_0, fold_1 = self.split_kfold(mmap)
        fold_0.to_csv(fpath_fold0, index=False)
        fold_1.to_csv(fpath_fold1, index=False)
        fold_0_label_0_ratio = 1 - (fold_0['label'].sum() / fold_0.shape[0])
        print(f"label 0 ratio: {fold_0_label_0_ratio}")

    def get_case_id(self, slide_id):
        return re.match(r"([^_]+)", slide_id).group(1)
    
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