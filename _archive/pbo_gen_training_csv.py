
import re
import pandas as pd

dataroot = "../CLAM/"
path = {
    'dpath_features': f"{dataroot}FEATURES_DIRECTORY/pt_files/",
    'fpath_map': f"{dataroot}dataset_csv/extract_features.csv",
    "fpath_out":f"{dataroot}dataset_csv/training_map.csv"
}

df = pd.read_csv(path['fpath_map'])
print(df)

new_df = []
for _, row in df.iterrows():
    ext_slide_id = row['slide_id']
    case_patient_id = re.match(r"(patient_[^_]+)", ext_slide_id).group(1)
    trn_slide_id = f"{ext_slide_id.rstrip('.mrxs')}"
    label = ['neg', 'pos'][row['label']]
    new_df.append({
        'case_id':case_patient_id,
        'slide_id':trn_slide_id,
        'label':label,
    })

training_map = pd.DataFrame(new_df)
training_map = training_map.sample(frac=1)
print(training_map)
training_map.to_csv(path["fpath_out"], index=False)

