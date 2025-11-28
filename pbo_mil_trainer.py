
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime

def plot_performance(protocol, runpath):
    epochs = range(1, len(protocol.valcost) + 1)
    traincol_0 = 'tab:blue'
    testcol_0 = 'tab:red'
    import matplotlib.pyplot as plt
    plt.plot(epochs, protocol.traincost, traincol_0, label='train_0')
    plt.plot(epochs, protocol.valcost, testcol_0, label='val_0')
    if False:
        traincol_1 = 'tab:orange'
        testcol_1 = 'tab:brown'
        plt.plot(epochs, protocol.traincost_1, traincol_1, label='train_1')
        plt.plot(epochs, protocol.valcost_1, testcol_1, label='val_1')
    #plt.ylim([0, 1.2*protocol.traincost_0[0]])
    plt.legend()
    plt.ylabel('Cost')
    plt.tight_layout()
    plt.savefig(f'{runpath}performance.png')
    plt.figure()
    plt.close('all')


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
        label = torch.tensor(row['label'].item(), dtype=torch.long)
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
            logits, _, Y_hat, results_dict = self.model(features, labels, True)
            raw_loss = self.criterion(logits, labels)
            loss = 0.7 * raw_loss + (1 - 0.7) * results_dict['instance_loss']
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
        with torch.inference_mode():
            for features, labels in valloader:
                features, labels = features.squeeze(0).to(device), labels.to(device)
                logits, _, _, _ = self.model(
                    features,
                    label=labels,
                    instance_eval=True,
                )
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
        self.runpath = self.init_run(hparam, augm, tag, device)
        self.trainer = self.learn_parameters(
            runpath=self.runpath,
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
        return trainer