
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime

"""
class MIL_fc(nn.Module):
    # modified from models/model_mil/MIL_fc
    def __init__(self, dropout):
        super().__init__()
        embed_dim = 1024
        size = [embed_dim, 512]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifier =  nn.Linear(size[1], 2)
        self.top_k = 1

    def forward(self, h):
        print(h)
        #raise SystemExit
        h = self.fc(h)
        logits  = self.classifier(h) # K x 2
        
        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1)

        return top_instance, Y_prob, Y_hat, y_probs
"""

class MILReader(Dataset):
    def __init__(self, fpath_map, dpath_features_pt, mode):
        assert mode in ['train', 'test']
        self.mode = mode
        self.dpath_features_pt = dpath_features_pt
        self.map = pd.read_csv(fpath_map)
    
    def __len__(self):
        return self.map.shape[0]
    
    def __getitem__(self, idx):
        row = self.map.iloc[idx]
        slide_id = row['slide_id']
        fpath_raw_features = os.path.join(self.dpath_features_pt, f"{slide_id}.pt")
        features = torch.load(fpath_raw_features)
        label = torch.tensor([row['label'].item()], dtype=torch.float32)
        #print(features.shape)
        if self.mode == 'train':
            return features, label
        else:
            return slide_id, features, label


class MILTrainer:
    def __init__(self, model, lr, weight_decay, device):
        #from topk.svm import SmoothTop1SVM
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.traincost, self.valcost = [], []
        self.trainperformance, self.valperformance = [], []
        self.current_best = None
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def get_nr_accurate(self, logits, labels):
        logits = logits.argmax(1)
        accurate_pred = (logits == labels)
        nr_accurate = accurate_pred.sum()
        return nr_accurate.item()

    def train_epoch(self, trainloader, device):
        self.model.train()
        cost, performance = 0, 0
        for features, labels in trainloader:
            features, labels = features.squeeze(0).to(device), labels.to(device)
            logits, Y_prob, Y_hat, A_raw, results_dict = self.model(features)
            print(logits)
            print(Y_prob)
            print(Y_hat)
            print(A_raw)
            print(results_dict)
            raise SystemExit
        
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            cost += loss.item() / len(trainloader)
            performance += self.get_nr_accurate(logits, labels) / len(trainloader.dataset)
        self.traincost += [cost]
        self.trainperformance += [performance]

    def val_epoch(self, valloader, device):
        self.model.eval()
        cost, performance = 0, 0
        with torch.no_grad():
            for features, labels in valloader:
                features, labels = features.to(device), labels.to(device)
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                cost += loss.item() / len(valloader)
                performance += self.get_nr_accurate(logits, labels) / len(valloader.dataset)
        self.valcost += [cost]
        self.valperformance += [performance]

    def log_epoch(self, header, runpath, nr_epochs):
        epoch_info = f'{len(self.valcost)}/{nr_epochs}\t{round(self.traincost[-1], 4)}/{round(self.valcost[-1], 4)}\t{round(self.trainperformance[-1], 4)}/{round(self.valperformance[-1], 4)}\t{str(datetime.now())[11:19]}'
        if self.current_best is None or self.current_best >= self.valcost[-1]: # early stopping protocol
            self.current_best = self.valcost[-1]
            path_model = f"{runpath}model.pth"
            record_history(path_model)
            torch.save(self.model.state_dict(), path_model)
            epoch_info = f'{epoch_info} saved!'
        print(epoch_info)
        with open(runpath + 'log.txt', 'a') as file:
            if len(self.valcost) <= 1:
                file.write(header+'\n')
            file.write(epoch_info+'\n')
        
    def execute_train_protocol(self, trainloader, valloader, nr_epochs, runpath, device):
        print(f'\nbeginning training {str(datetime.now())[11:19]}')
        header = f'epoch\tloss\taccuracy\ttime'
        print(header)
        for i in range(1, nr_epochs+1):
            self.train_epoch(trainloader, device)
            self.val_epoch(valloader, device)
            self.log_epoch(header, runpath, nr_epochs)

def record_history(path_model):
    if not os.path.exists(path_model):
        with open(f'run/history/history.txt', 'a') as file:
            file.write(f"\n{path_model}")


class MilTrainWrapper:
    def __init__(self, dpath_features_pt, fpath_map_fold_0,
        fpath_map_fold_1, hparam, state_dict, augm, tag, device):

        loader_0, loader_1 = self.init_loaders(
            dpath_features_pt=dpath_features_pt,
            fpath_map_fold_0=fpath_map_fold_0,
            fpath_map_fold_1=fpath_map_fold_1,
            batch_size=hparam['batch_size'],
        )
        
        model = self.init_model(hparam['dropout'], device)
        runpath = self.init_run(hparam, augm, tag, device)
        self.learn_parameters(
            runpath=runpath,
            loader_0=loader_0,
            loader_1=loader_1,
            model=model,
            nr_epochs=hparam['nr_epochs'],
            lr=hparam['learning_rate'],
            weight_decay=hparam['weight_decay'],
            device=device,
        )


    def init_loaders(self, dpath_features_pt, fpath_map_fold_0, fpath_map_fold_1, batch_size):
        print('initiating loaders ...')
        reader_0 = MILReader(fpath_map_fold_0, dpath_features_pt, 'train')
        reader_1 = MILReader(fpath_map_fold_1, dpath_features_pt, 'train')
        
        from torch.utils.data import DataLoader
        loader_0 = DataLoader(reader_0, batch_size, shuffle=True)
        loader_1 = DataLoader(reader_1, batch_size, shuffle=True)
        return loader_0, loader_1

    def init_run(self, hparam, augmentation, tag, device):
        current = datetime.now()
        runpath = f'run/{tag}/{str(current)[8:10]}_{str(current)[11:13]}_{str(current)[14:16]}_{str(current)[17:19]}/'
        # create dir for run history if none exists
        if not os.path.exists(f'run/history/'):
            os.makedirs(f'run/history/')
        # create directory for output files of current run
        if not os.path.isdir(runpath):
            os.makedirs(runpath)
        # saving runlog to current run folder
        with open(os.path.join(runpath, 'log.txt'), 'a') as file:
            file.write('################# initiating run #################\n')
            file.write(f'run dir has been created in {runpath}\n\n')
            for param in hparam:
                file.write(f'{param}\t: {hparam[param]}\n')
            for aug in augmentation:
                file.write(f'{aug}\t: {augmentation[aug]}\n')
            file.write('\n')
            file.write(f'Using {device} device\n')
            file.write('################# initiated run ##################\n')
        # printing current state of logfile to terminal
        with open(runpath+'log.txt', 'r') as file:
            for_terminal = file.read()
        print(for_terminal)
        return runpath

    def init_model(self, dropout, device):
        print('initiating model ...')
        from pbo_mil_model import CLAM_SB
        model = CLAM_SB(dropout)
        return model.to(device)
    
    def learn_parameters(self, runpath, loader_0, loader_1, model,
        nr_epochs, lr, weight_decay, device):
        print('init training ...')
        trainer = MILTrainer(model, lr, weight_decay, device)
        trainer.execute_train_protocol(
            trainloader=loader_0,
            valloader=loader_1,
            nr_epochs=nr_epochs,
            runpath=runpath,
            device=device,
        )
        