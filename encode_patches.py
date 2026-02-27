
import torch
import torchvision
from torch.utils.data import DataLoader
import h5py
import openslide
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils.file_utils import save_hdf5
from torch.utils.data import Dataset
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms

def map_slide2patient(x, slide_id):
    pattern = slide_id.split('_')[0]
    return x.lstrip('patient_') == pattern

class WSI2BagsReader(Dataset):
    def __init__(self, fpath_qualityLog, wsi, img_transforms, fltr_params):
        self.map = self.apply_filter(fpath_qualityLog, fltr_params)
        self.patch_lvl = self.map['patch_lvl'].unique().item()
        self.patch_size = self.map['patch_size'].unique().item()
        self.wsi = wsi
        self.roi_transforms = img_transforms
        print(self.roi_transforms)

    def apply_filter(self, fpath_qualityLog, fltr_params):
        qlog = pd.read_csv(fpath_qualityLog)
        qlog = qlog[qlog['bg'] <= fltr_params['bg']]
        qlog = qlog[qlog['blur'] >= fltr_params['blur']]
        qlog = qlog.sample(frac=1) # important for random origin of patch on slide distribution into bags
        return qlog
            
    def __len__(self):
        return self.map.shape[0]

    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        coord = np.array((row['pos_x'], row['pos_y']), dtype=np.int32)
        patch = self.wsi.read_region(coord, self.patch_lvl, (self.patch_size, self.patch_size)).convert('RGB')
        

        #import matplotlib.pyplot as plt
        #plt.imshow(patch)
        #plt.tight_layout()
        #plt.show()
        
        patch = self.roi_transforms(patch)
        return {'patch': patch, 'coord': coord}


class FeatureX:
    def __init__(
        self, dpath_qualityLog, dpath_wsiRoot, dpath_ptFeature, dpath_h5Feature,
        fpath_segmlog, fpath_encodingMap, fpath_Xmodel, fpath_patientInfo,
        batch_size, patch_size, fltr_params, target_bag_size,
    ):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = self.get_pbo_encoder(fpath_Xmodel)
        constants = MODEL2CONSTANTS['resnet50_trunc']
        img_transforms = get_eval_transforms(
            mean=constants['mean'],
            std=constants['std'],
            target_img_size=patch_size,
        )
        
        model.eval()
        model = model.to(device)

        loader_kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == "cuda" else {}

        if fpath_encodingMap.is_file():
            encode_prgs = pd.read_csv(fpath_encodingMap)
        else:
            encode_prgs = None

        patient_info = pd.read_csv(fpath_patientInfo)
        slide_ids = pd.read_csv(fpath_segmlog)['slide_id'].unique().tolist()

        for slide_id in tqdm(slide_ids):
            fpath_wsi = dpath_wsiRoot / f"patient_{slide_id}.mrxs"

            if encode_prgs is not None:
                handled_slides = encode_prgs['slide_id'].unique().tolist()
                if slide_id in handled_slides:
                    print(f'already handled {slide_id}')
                    continue 

            fpath_qualityLog = dpath_qualityLog / f"{slide_id}.csv"
            wsi = openslide.open_slide(fpath_wsi)
            dataset = WSI2BagsReader(
                fpath_qualityLog=fpath_qualityLog,
                wsi=wsi,
                img_transforms=img_transforms,
                fltr_params=fltr_params,
            )

            loader = DataLoader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
            fpaths_h5Feature = self.compute_w_loader(device, dpath_h5Feature, loader, model, slide_id, target_bag_size)

            to_append = []
            condition = patient_info['patient_n'].apply(map_slide2patient, args=(slide_id,))
            isup = patient_info.loc[condition, 'isup'].item()
            label = int(isup > 1)
            age = patient_info.loc[condition, 'age'].item()
            psa = patient_info.loc[condition, 'psa'].item()
            for fpath in fpaths_h5Feature:
                bag_id = fpath.name.removesuffix('.h5')
                with h5py.File(fpath, "r") as file:
                    features = file['features']
                    dim0, dim1 = features.shape

                to_append.append({
                    'slide_id':slide_id,
                    'bag_id':bag_id,
                    'label':label,
                    'isup':isup,
                    'age':age,
                    'psa':psa,
                    'dim0':dim0,
                    'dim1':dim1,
                })

            appendix = pd.DataFrame(to_append)
            if encode_prgs is None:
                encode_prgs = appendix
            else:
                encode_prgs = pd.concat([encode_prgs, appendix], axis=0)
            encode_prgs.to_csv(fpath_encodingMap, index=False)
        
        self.convert_h52pt(dpath_ptFeature, dpath_h5Feature)
    
    def convert_h52pt(self, dpath_ptFeature, dpath_h5Feature):
        fpaths = list(dpath_h5Feature.iterdir())
        for fpath in tqdm(fpaths):
            bag_id = fpath.name.removesuffix('.h5')
            with h5py.File(fpath, "r") as file:
                features = file['features'][:]
                torch.save(
                    torch.from_numpy(features),
                    dpath_ptFeature / f"{bag_id}.pt",
                )

    def compute_w_loader(self, device, dpath_h5Feature, loader, model, slide_id, target_bag_size):
        nr_splits = max(len(loader.dataset) // target_bag_size, 1)
        fpaths_h5Feature = [dpath_h5Feature / f"{slide_id}_{i}.h5" for i in range(nr_splits)]

        for data in loader:
            with torch.inference_mode():

                batch = data['patch']
                coords = data['coord']

                coords = np.array(coords, dtype=np.int32)
                batch = batch.to(device, non_blocking=True)

                features = model(batch)
                features = features.cpu().numpy().astype(np.float32)

                asset_dict = {'features': features, 'coords': coords}
                fpath_h5Feature = np.random.choice(fpaths_h5Feature)
                save_hdf5(
                    output_path=fpath_h5Feature,
                    asset_dict=asset_dict,
                    attr_dict=None,
                    mode=['w', 'a'][fpath_h5Feature.is_file()]
                )
        return fpaths_h5Feature

    def get_pbo_encoder(self, fpath_Xmodel):
        def process_state_dict(state_dict):
            for k in list(state_dict.keys()):
                state_dict[k.replace("model.", "").replace("resnet.", "")] = state_dict.pop(k)
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("fc.")
            }
            return state_dict

        model = torchvision.models.resnet18(weights=None)
        model.fc = torch.nn.Identity()
        state = torch.load(fpath_Xmodel, map_location='cuda:0', weights_only=False)
        state_dict = process_state_dict(state["state_dict"])
        model.load_state_dict(state_dict, strict=True)
        return model
