import re
import numpy as np
import pandas as pd

class FeatureMapSplitter:
    def __init__(self, fpath_encodingMap, dpath_milFolds):
        mmap = pd.read_csv(fpath_encodingMap)
        mmap['case_id'] = mmap['slide_id'].apply(self.get_case_id)
        self.split_kfold(mmap, dpath_milFolds)

    def get_case_id(self, slide_id):
        return re.match(r"([^_]+)", slide_id).group(1)
    
    def split_kfold(self, mmap, dpath_milFolds, k=6):
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
            k=k,
        )

        keep_cols = [col for col in fmap.columns if not any(col.startswith(char) for char in ['mod_', 'age_', 'isup_'])]
        fmap = fmap[keep_cols]
        for _, row in fmap.iterrows():
            case_id = row['case_id']
            fold_id = row['fold_id']
            mmap.loc[mmap['case_id']==case_id, 'fold_id'] = fold_id

        fold_indices = fmap['fold_id'].unique()
        for idx in fold_indices:
            fpath_fold = dpath_milFolds / f'fold_{idx}.csv'
            fold = mmap[mmap['fold_id'] == idx]
            fold = fold.sample(frac=1)
            fold = fold.reset_index(drop=True)
            fold = fold.drop(columns=['fold_id'])
            fold.to_csv(fpath_fold, index=False)
            print(f"fold_{idx}| psa: {fold['psa'].std()}, isup: {fold['isup'].std()}")
