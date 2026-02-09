
""""
continue when i know what info the output file needs
"""


import pandas as pd


class SlideMapWriter:
    def __init__(self, dpath_wsiRoot, fpath_segmParam, fpath_patientInfo):
        ids_from_dir = [
            name.lstrip('patient_').rstrip('.mrxs')
            for name in [
                fpath.name for fpath in dpath_wsiRoot.iterdir()
                if fpath.name.endswith('.mrxs')
            ]
        ]

        df_patientInfo = pd.read_csv(fpath_patientInfo)
        df_segmParam = pd.read_csv(fpath_segmParam)
        assert df_segmParam.shape[0] == df_segmParam['slide_id'].unique().shape[0]


        for _, row in df_segmParam.iterrows():
            assert row['category'] == 1, f"{row['category']}, {type(row['category'])}"
            slide_id = row['slide_id']



        print(ids_from_dir)
        print(len(ids_from_dir))

from config import *
smw = SlideMapWriter(
    cfg.dpath_wsiRoot,
    cfg.fpath_segmParam,
    cfg.fpath_patientInfo,
)
        
