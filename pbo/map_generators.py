
import os
import pandas as pd

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

