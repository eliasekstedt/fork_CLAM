
import os
import pandas as pd
import re

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
                    'label':row['positive']
                })
        return pd.DataFrame(patchset_map)

class ClassifierMapGenerator:
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
        wedge = len(patient_ids) // 2
        fold_0_patient_ids = patient_ids[:wedge]
        fold_1_patient_ids = patient_ids[wedge:]
        map_fold_0 = map_classifier[map_classifier['case_id'].isin(fold_0_patient_ids)]
        map_fold_1 = map_classifier[map_classifier['case_id'].isin(fold_1_patient_ids)]

        assert map_fold_0.shape[0] + map_fold_1.shape[0] == map_classifier.shape[0]
        assert map_fold_0[map_fold_0['case_id'].isin(fold_1_patient_ids)].shape[0] == 0
        assert map_fold_1[map_fold_1['case_id'].isin(fold_0_patient_ids)].shape[0] == 0

        return map_fold_0, map_fold_1
