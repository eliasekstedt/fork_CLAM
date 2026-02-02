
import os
import pandas as pd
import re
import numpy as np

class PatchsetMapGenerator:
    def __init__(self, dpath_mrxs, dpath_patchset, fpath_map_patient, fpath_map_patchset):
        patchset_ids = [
            id.rstrip('.h5')
            for id in os.listdir(dpath_patchset)
            if id.endswith('.h5')
        ]
        slide_ids = [
            id for id in os.listdir(dpath_mrxs)
            if id.endswith('.mrxs')
            and id.rstrip('.mrxs') in patchset_ids
        ]

        patient_info = pd.read_csv(fpath_map_patient)
        patchset_map = self.create_patchset_map(patient_info, slide_ids)
        patchset_map['slide_id'] = patchset_map['patchset_id'].str.rstrip('.mrxs')
        print(patchset_map)
        print(patchset_map.shape)
        patchset_map.to_csv(fpath_map_patchset, index=False)

    def create_patchset_map(self, patient_info, slide_ids):
        patchset_map = []
        for _, row in patient_info.iterrows():
            patient_id = row['patient_n']
            
            this_patients_slide_ids = [
                id for id in slide_ids
                if patient_id in id
            ]

            for id in this_patients_slide_ids:
                patchset_map.append({
                    'patchset_id':id,
                    'age':row['age'],
                    'psa':row['psa'],
                    'isup':row['isup'],
                    'label':int(row['isup'] > 1),
                })
        return pd.DataFrame(patchset_map)

class ClassifierMapGenerator:
    def __init__(self, fpath_map_patchset, fpath_map_fold_0, fpath_map_fold_1):
        mmap = pd.read_csv(fpath_map_patchset)
        mmap['case_id'] = mmap['patchset_id'].apply(self.get_case_id)
        map_fold_0, map_fold_1 = self.split_kfold(mmap)

        map_fold_0.to_csv(fpath_map_fold_0, index=False)
        map_fold_1.to_csv(fpath_map_fold_1, index=False)
        
        fold_0_label_0_ratio = 1 - (map_fold_0['label'].sum() / map_fold_0.shape[0])
        print(f"label 0 ratio: {fold_0_label_0_ratio}")

    def get_case_id(self, patchset_id):
        return re.match(r"(patient_[^_]+)", patchset_id).group(1)
    
    def split_kfold(self, mmap, k=5):
        def score_fold(fold, dcol_names):
            as_array = fold[dcol_names].to_numpy()
            return np.sum(np.sum(as_array, axis=0) ** 2).item()

        def create_balanced_splits(df, jcol_name, dcol_names, k):
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
        
        jcol_name = 'slide_id'
        fmap = create_balanced_splits(
            df=dmap[[jcol_name] + dcol_names],
            jcol_name=jcol_name,
            dcol_names=dcol_names,
            k=5,
        )
        keep_cols = [col for col in fmap.columns if not any(col.startswith(char) for char in ['mod_', 'age_', 'isup_'])]
        fmap = fmap[keep_cols]
        fmap = mmap.merge(fmap, on='slide_id')
        fold_0 = fmap[fmap['fold_id'] < 4]
        fold_1 = fmap[~fmap['fold_id'].isin(fold_0['fold_id'])]
        fold_0 = fold_0.drop(columns=['fold_id'])
        fold_1 = fold_1.drop(columns=['fold_id'])
        print(fold_0['psa'].std(), fold_1['psa'].std())
        print(fold_0['isup'].std(), fold_1['isup'].std())
        return fold_0, fold_1

class ObsoleteClassifierMapGenerator:
    def __init__(self, fpath_map_patchset, fpath_map_fold_0, fpath_map_fold_1):
        patchset_map = pd.read_csv(fpath_map_patchset)
        patchset_map['case_id'] = patchset_map['patchset_id'].apply(self.get_case_id)
        map_classifier = patchset_map[['case_id', 'slide_id', 'label']]
        map_fold_0, map_fold_1 = self.split_2fold(map_classifier)
        map_fold_0.to_csv(fpath_map_fold_0, index=False)
        map_fold_1.to_csv(fpath_map_fold_1, index=False)
        fold_0_label_0_ratio = 1 - (map_fold_0['label'].sum() / map_fold_0.shape[0])
        print(f"label 0 ratio: {fold_0_label_0_ratio}")

    def get_case_id(self, patchset_id):
        return re.match(r"(patient_[^_]+)", patchset_id).group(1)
    
    def split_2fold(self, map_classifier):
        map_classifier = map_classifier.sample(frac=1)
        patient_ids = map_classifier['case_id'].unique()
        wedge = int(len(patient_ids) * 0.8)
        fold_0_patient_ids = patient_ids[:wedge]
        fold_1_patient_ids = patient_ids[wedge:]
        map_fold_0 = map_classifier[map_classifier['case_id'].isin(fold_0_patient_ids)]
        map_fold_1 = map_classifier[map_classifier['case_id'].isin(fold_1_patient_ids)]

        assert map_fold_0.shape[0] + map_fold_1.shape[0] == map_classifier.shape[0]
        assert map_fold_0[map_fold_0['case_id'].isin(fold_1_patient_ids)].shape[0] == 0
        assert map_fold_1[map_fold_1['case_id'].isin(fold_0_patient_ids)].shape[0] == 0

        return map_fold_0, map_fold_1


from pbo_config import *
PatchsetMapGenerator(
    dpath_mrxs=cfg.dpath_mrxsRoot,
    dpath_patchset=cfg.dpath_patchset,
    fpath_map_patient=cfg.fpath_map_patient_info,
    fpath_map_patchset=cfg.fpath_map_patchset,
)