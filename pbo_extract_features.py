import time
import os
import torch
import torchvision
from torch.utils.data import DataLoader
import h5py
import openslide
from tqdm import tqdm
import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
#from models import get_encoder
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms

class FeatureExtractor:
    def __init__(
        self, dpath_patchset, dpath_mrxsRoot, dpath_features_pt, dpath_features_h5,
        fpath_map_patchset, fpath_model, batch_size, patch_size, slide_extension,
        no_auto_skip,
    ):
        self.dpath_patchset = dpath_patchset
        self.dpath_mrxsRoot = dpath_mrxsRoot
        self.dpath_features_pt = dpath_features_pt
        self.dpath_features_h5 = dpath_features_h5
        self.fpath_map_patchset = fpath_map_patchset
        self.fpath_model = fpath_model
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.slide_extension = slide_extension
        self.no_auto_skip = no_auto_skip
    
    def __call__(self):
        print('initializing dataset')
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if self.fpath_map_patchset is None:
            raise NotImplementedError

        bags_dataset = Dataset_All_Bags(self.fpath_map_patchset)

        dest_files = os.listdir(self.dpath_features_pt)

        model = self.get_pbo_encoder(self.fpath_model)

        constants = MODEL2CONSTANTS['pbo_subs']
        img_transforms = get_eval_transforms(
            mean=constants['mean'],
            std=constants['std'],
            target_img_size=self.patch_size,
        )
                
        _ = model.eval()
        model = model.to(device)
        total = len(bags_dataset)

        loader_kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == "cuda" else {}
        
        #with open('slidelog.txt', "r") as f:
        #    previously = [line.strip() for line in f]

        for bag_candidate_idx in tqdm(range(total)):
            slide_id = bags_dataset[bag_candidate_idx].split(self.slide_extension)[0]
            #if slide_id in previously:
            #    print(f"skipping")
            #    continue

            bag_name = slide_id + '.h5'

            with open('slidelog.txt', 'a') as file:
                file.write(f"{slide_id}\n")
            #print(slide_id)

            h5_file_path = os.path.join(self.dpath_patchset, bag_name)
            slide_file_path = os.path.join(self.dpath_mrxsRoot, slide_id+self.slide_extension)
            #print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
            print(slide_id)

            """
            """
            if not self.no_auto_skip and slide_id + '.pt' in dest_files:
                print('skipped {}'.format(slide_id))
                continue 
            
            #print(slide_id)
            output_path = os.path.join(self.dpath_features_h5, bag_name)
            time_start = time.time()
            wsi = openslide.open_slide(slide_file_path)
            dataset = Whole_Slide_Bag_FP(
                file_path=h5_file_path, 
                wsi=wsi, 
                img_transforms=img_transforms,
            )

            loader = DataLoader(dataset=dataset, batch_size=self.batch_size, **loader_kwargs)
            output_file_path = self.compute_w_loader(device, output_path, loader, model)

            time_elapsed = time.time() - time_start
            print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

            with h5py.File(output_file_path, "r") as file:
                features = file['features'][:]
                print('features size: ', features.shape)
                print('coordinates size: ', file['coords'].shape)

            features = torch.from_numpy(features)
            bag_base, _ = os.path.splitext(bag_name)
            torch.save(features, os.path.join(self.dpath_features_pt, bag_base+'.pt'))
    
    def get_pbo_encoder(self, fpath_model):
        """
        this function is code modified from:
        https://github.com/ozanciga/self-supervised-histopathology?tab=readme-ov-file
        the .ckpt file can also be downloaded from there.
        """
        def load_model_weights(model, weights):
            model_dict = model.state_dict()
            weights = {k: v for k, v in weights.items() if k in model_dict}
            if weights == {}:
                print('No weight could be loaded..')
            model_dict.update(weights)
            model.load_state_dict(model_dict)
            return model

        model = torchvision.models.__dict__['resnet18'](weights=None)
        state = torch.load(fpath_model, map_location='cuda:0', weights_only=False)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        model = load_model_weights(model, state_dict)
        model.fc = torch.nn.Sequential()
        #model = model.cuda()
        return model

    def compute_w_loader(self, device, output_path, loader, model):
        """
        args:
            output_path: directory to save computed features (.h5 file)
            model: pytorch model
            verbose: level of feedback
        """
        if True:
            print(f'processing a total of {len(loader)} batches'.format(len(loader)))

        mode = 'w'
        for count, data in enumerate(tqdm(loader)):
            with torch.inference_mode():	
                batch = data['img']
                coords = data['coord'].numpy().astype(np.int32)
                batch = batch.to(device, non_blocking=True)
                
                features = model(batch)
                features = features.cpu().numpy().astype(np.float32)

                asset_dict = {'features': features, 'coords': coords}
                save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
                mode = 'a'
        
        return output_path


