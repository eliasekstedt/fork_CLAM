import time
import os
import torch
import torchvision
from torch.utils.data import DataLoader
import h5py
import openslide
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Whole_Slide_Bag_FP, Dataset_All_Bags
#from models import get_encoder
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms

class FeatureX:
    def __init__(
        self, dpath_qualityLog, dpath_wsiRoot, dpath_ptFeature, dpath_h5Feature,
        fpath_segmlog, fpath_encodingPrgs, fpath_encodingMap, fpath_Xmodel, fpath_patientInfo,
        batch_size, patch_size, fltr_params,
    ):
        self.dpath_qualityLog = dpath_qualityLog
        self.dpath_wsiRoot = dpath_wsiRoot
        self.fpath_Xmodel = fpath_Xmodel
        self.fpath_patientInfo = fpath_patientInfo
        self.batch_size = batch_size
        self.patch_size = patch_size

        self.dpath_ptFeature = dpath_ptFeature
        self.dpath_h5Feature = dpath_h5Feature
        self.fpath_encodingPrgs = fpath_encodingPrgs
        self.fpath_encodingMap = fpath_encodingMap

        self.fpath_segmlog = fpath_segmlog
        self.fltr_params = fltr_params
        
        
        
        
        
    
    def __call__(self):
        print('initializing dataset')
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = self.get_pbo_encoder(self.fpath_Xmodel)
        constants = MODEL2CONSTANTS['pbo_subs']
        img_transforms = get_eval_transforms(
            mean=constants['mean'],
            std=constants['std'],
            target_img_size=self.patch_size,
        )
        
        model.eval()
        model = model.to(device)

        loader_kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == "cuda" else {}

        if self.fpath_encodingPrgs.is_file():
            slide_prgs = pd.read_csv(self.fpath_encodingPrgs)
        else:
            slide_prgs = pd.read_csv(self.fpath_segmlog)[['slide_id', 'handled']]
            slide_prgs['handled'] = 0

        if self.fpath_encodingMap.is_file():
            encode_prgs = pd.read_csv(self.fpath_encodingMap)
        else:
            encode_prgs = None

        patient_info = pd.read_csv(self.fpath_patientInfo)
        for _, row in tqdm(slide_prgs.iterrows()):
            slide_id = row['slide_id']
            fpath_wsi = self.dpath_wsiRoot / f"patient_{slide_id}.mrxs"

            if row['handled'] == 1:
                print(f'already handled {slide_id}')
                continue 

            #fpath_h5Feature = self.dpath_h5Feature / f"{slide_id}.h5"
            fpath_qualityLog = self.dpath_qualityLog / f"{slide_id}.csv"
            wsi = openslide.open_slide(fpath_wsi)
            dataset = Whole_Slide_Bag_FP(
                fpath_qualityLog=fpath_qualityLog,
                wsi=wsi,
                img_transforms=img_transforms,
                fltr_params=self.fltr_params,
            )

            loader = DataLoader(dataset=dataset, batch_size=self.batch_size, **loader_kwargs)
            fpaths_h5Feature = self.compute_w_loader(device, self.dpath_h5Feature, loader, model, slide_id)

            to_append = []
            condition = patient_info['patient_n'].rstrip('patient_') == str(slide_id.split('_')[0])
            isup = patient_info.loc[condition, 'isup']
            label = int(isup > 1)
            age = patient_info.loc[condition, 'age']
            psa = patient_info.loc[condition, 'psa']
            for fpath in fpaths_h5Feature:
                bag_id = fpath.name.rstrip('.h5')
                with h5py.File(fpath, "r") as file:
                    features = file['features'][:]
                    print('features size: ', features.shape)
                    print('coordinates size: ', file['coords'].shape)

                print(fpath, features.shape)
                torch.save(
                    torch.from_numpy(features),
                    self.dpath_ptFeature / f"{bag_id}.pt",
                )

                to_append.append({
                    'slide_id':slide_id,
                    'bag_id':bag_id,
                    'label':label,
                    'isup':isup,
                    'age':age,
                    'psa':psa,
                })

            encode_prgs_appendix = pd.DataFrame(to_append)
            if encode_prgs is None:
                encode_prgs = encode_prgs
            else:
                encode_prgs = pd.concat([encode_prgs, encode_prgs_appendix], axis=0)
            encode_prgs.to_csv(self.fpath_encodingMap, index=False)

            slide_prgs.loc[slide_prgs['slide_id']==slide_id, 'handled'] = 1
            slide_prgs.to_csv(self.fpath_encodingPrgs, index=False)
            raise SystemExit

    def compute_w_loader(self, device, dpath_h5Feature, loader, model, slide_id):
        nr_patches = loader.dataset.map.shape[0]
        nr_splits = max(len(loader.dataset) // 1000, 1)
        fpaths_h5Feature = [dpath_h5Feature / f"{slide_id}_{i}.h5" for i in range(nr_splits)]
        print(nr_patches, nr_splits)

        for data in tqdm(loader):
            with torch.inference_mode():

                batch = data['img']
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
