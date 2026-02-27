
import os
import pandas as pd
#import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datetime import datetime

def plot_performance(protocol, runpath):
    epochs = range(1, len(protocol.valcost) + 1)
    import matplotlib.pyplot as plt
    for train_item, val_item, name in zip(
        [protocol.traincost, protocol.trainperformance, protocol.train_tp, protocol.train_pp],
        [protocol.valcost, protocol.valperformance, protocol.val_tp, protocol.val_pp],
        ['cost', 'accuracy', 'true_positives', 'predicted_positives'],
    ):
        plt.plot(epochs, train_item, 'tab:blue', label='train_0')
        plt.plot(epochs, val_item, 'tab:red', label='val_0')
        plt.ylim([0, max(train_item + val_item + [1])]) # crude
        plt.legend()
        plt.ylabel(name)
        plt.xlabel('Epoch')
        plt.tight_layout()
        plt.savefig(f'{runpath}{name}.png')
        plt.figure()
        plt.close('all')

def file_it(file_name, message, to_terminal=False):
    if to_terminal:
        print(message)
    with open(file_name, 'a') as file:
        file.write(f'{message}\n')

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
        bag_id = row['bag_id']
        fpath_raw_features = os.path.join(self.dpath_features_pt, f"{bag_id}.pt")
        features = torch.load(fpath_raw_features)
        label = torch.tensor(row['label'].item(), dtype=torch.long)
        if self.mode == 'train':
            return features, label
        else:
            return bag_id, features, label

class MILTrainer:
    def __init__(self, model, lr, lf_weights, weight_decay, train_ap, val_ap, device):
        self.criterion = nn.CrossEntropyLoss(weight=lf_weights).to(device)
        self.train_ap, self.val_ap = train_ap, val_ap

        self.traincost, self.trainperformance = [], []
        self.train_tp, self.train_pp = [], []
        self.valcost, self.valperformance = [], []
        self.val_tp, self.val_pp = [], []
        self.current_best = None

        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def get_nr_accurate(self, logits, labels):
        logits = logits.argmax(1)
        accurate_pred = (logits == labels)
        nr_accurate = accurate_pred.sum()
        return nr_accurate.item()

    def train_epoch(self, trainloader, device):
        cost, performance, tp, pp = [0]*4
        self.model.train()
        for features, label in trainloader:
            features, label = features.squeeze(0).to(device), label.to(device)
            logit, _, Y_hat, results_dict = self.model(features, label, True)

            pred_item = logit.argmax(1).item()
            label_item = label.item()
            if all([item == 1 for item in [pred_item, label_item]]):
                tp += 1
            if pred_item == 1:
                pp += 1

            raw_loss = self.criterion(logit, label)
            loss = 0.7 * raw_loss + (1 - 0.7) * results_dict['instance_loss']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            cost += loss.item() / len(trainloader)
            performance += self.get_nr_accurate(logit, label) / len(trainloader.dataset)
        
        self.train_tp.append(tp)
        self.train_pp.append(pp)
        self.traincost += [cost]
        self.trainperformance += [performance]

    def val_epoch(self, valloader, device):
        cost, performance, tp, pp = [0] * 4

        self.model.eval()
        with torch.inference_mode():
            for features, label in valloader:
                features, label = features.squeeze(0).to(device), label.to(device)
                logit, _, _, _ = self.model(features, label=label, instance_eval=True)

                pred_item = logit.argmax(1).item()
                label_item = label.item()
                if all([item == 1 for item in [pred_item, label_item]]):
                    tp += 1
                if pred_item == 1:
                    pp += 1

                loss = self.criterion(logit, label)
                cost += loss.item() / len(valloader)
                performance += self.get_nr_accurate(logit, label) / len(valloader.dataset)

        self.val_tp.append(tp)
        self.val_pp.append(pp)
        self.valcost += [cost]
        self.valperformance += [performance]

    def log_epoch(self, header, runpath, nr_epochs):
        epoch_info = '{}/{}\t{}|{}\t{}|{}\t{}|{}     \t{}|{}     \t{}|{}     \t{}|{} \t{}'.format(
            len(self.valcost), nr_epochs, round(self.traincost[-1], 4), round(self.valcost[-1], 4),
            round(self.trainperformance[-1], 4), round(self.valperformance[-1], 4),
            self.train_pp[-1], self.val_pp[-1],
            self.train_tp[-1], self.val_tp[-1],
            round(self.train_tp[-1] / (self.train_pp[-1] + 1e-8), 2), # precision
            round(self.val_tp[-1] / (self.val_pp[-1] + 1e-8), 2),
            round(self.train_tp[-1] / (self.train_ap + 1e-8), 2), # recall
            round(self.val_tp[-1] / (self.val_ap + 1e-8), 2),
            str(datetime.now())[11:19],
        )
        
        #epoch_score = self.valcost[-1] * (1 - self.val_tp[-1] / (self.val_pp[-1] + 1e-8))
        epoch_score = 1 / ((self.val_pp[-1] - self.val_ap)**2 / self.val_ap + 1) * self.val_tp[-1] / (self.val_pp[-1] + 1e-8)
        #print('ap: {}\npp: {}\nr: {}\nscore: {}|{}'.format(self.val_ap, self.val_pp[-1], self.val_tp[-1] / (self.val_pp[-1] + 1e-8), epoch_score, self.current_best))
        if self.current_best is None or self.current_best <= epoch_score: #self.valcost[-1]: # early stopping protocol
            self.current_best = epoch_score
            path_model = f"{runpath}model.pth"
            record_history(path_model)
            torch.save(self.model.state_dict(), path_model)
            epoch_info = f'{epoch_info} saved!'

        #print(epoch_info)
        

        if len(self.valcost) <= 1:
            file_it(f'{runpath}log.txt', '\n' + header, False)
        file_it(f"{runpath}log.txt", epoch_info, True)
        
        """
        with open(runpath + 'log.txt', 'a') as file:
            if len(self.valcost) <= 1:
                file.write(header+'\n')
            file.write(epoch_info+'\n')
        """

        
    def execute_train_protocol(self, trainloader, valloader, nr_epochs, runpath, device):
        print(f'\nbeginning training {str(datetime.now())[11:19]}')
        header = f'epoch\tloss\t\taccuracy\tpred_pos\ttrue_pos\tprecision\trecall  \ttime'
        print(header)

        ##########
        strike = 0
        ##########

        for _ in range(1, nr_epochs+1):

            ###############
            if strike >= 7:
                file_it(f'{runpath}log.txt', '\nx_x beyond recovery x_x\n', True)
                break
            ###############

            self.train_epoch(trainloader, device)
            self.val_epoch(valloader, device)
            self.log_epoch(header, runpath, nr_epochs)

            ##################################
            if any([self.valcost[-1] > 3 * self.traincost[-1]]):
                strike += 1
            else:
                strike = 0
            ##################################

def record_history(path_model):
    if not os.path.exists(path_model):
        with open(f'run/history/history.txt', 'a') as file:
            file.write(f"\n{path_model}")


class MilTrainWrapper:
    def __init__(self, dpath_ptFeature, fpath_fold0,
        fpath_fold1, hparam, fpath_state_dict, tag, device
    ):
        loader_0, loader_1 = self.init_loaders(
            dpath_features_pt=dpath_ptFeature,
            fpath_map_fold_0=fpath_fold0,
            fpath_map_fold_1=fpath_fold1,
            batch_size=hparam['batch_size'],
        )

        # class rebalance
        counts_0, counts_1 = [torch.tensor(
            loader.dataset.map['label'].value_counts(),
            dtype=torch.float
        ) for loader in [loader_0, loader_1]]

        lf_weights = 1.0 / counts_0
        lf_weights = lf_weights / lf_weights.sum()

        train_actual_pos, val_actual_pos = counts_0[1].item(), counts_1[1].item()
        logmore = {
            'nr_+_train/val':loader_0.dataset.map['label'].value_counts().tolist(),
            'lf_weights':lf_weights.numpy(),
            'ap_train/val':f"{int(train_actual_pos)}/{int(val_actual_pos)}",
        }
        
        runpath = self.init_run(hparam, logmore, tag, device)
        model = self.init_model(runpath, fpath_state_dict, hparam['dropout'], device)
        trainer = self.learn_parameters(
            runpath=runpath,
            loader_0=loader_0,
            loader_1=loader_1,
            model=model,
            nr_epochs=hparam['nr_epochs'],
            lr=hparam['learning_rate'],
            lf_weights=lf_weights,
            weight_decay=hparam['weight_decay'],
            train_ap=train_actual_pos,
            val_ap=val_actual_pos,
            device=device,
        )

        plot_performance(trainer, runpath)

    def init_loaders(self, dpath_features_pt, fpath_map_fold_0, fpath_map_fold_1, batch_size):
        print('initiating loaders ...')
        reader_0 = MILReader(fpath_map_fold_0, dpath_features_pt, 'train')
        reader_1 = MILReader(fpath_map_fold_1, dpath_features_pt, 'train')
        
        from torch.utils.data import DataLoader
        loader_0 = DataLoader(reader_0, batch_size, shuffle=True)
        loader_1 = DataLoader(reader_1, batch_size, shuffle=True)
        return loader_0, loader_1

    def init_run(self, hparam, logmore, tag, device):
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
            for key, val in {**hparam, **logmore}.items():
                file.write(f'{key}\t: {val}\n')
            file.write('\n')
            file.write(f'Using {device} device\n')
            file.write('################# initiated run ##################\n')
        # printing current state of logfile to terminal
        with open(runpath+'log.txt', 'r') as file:
            for_terminal = file.read()
        print(for_terminal)
        return runpath

    def init_model(self, runpath, fpath_state_dict, dropout, device):
        print('initiating model ...')
        from pbo_mil_model import CLAM_SB
        model = CLAM_SB(dropout)

        if not fpath_state_dict == '':
            model.load_state_dict(torch.load(fpath_state_dict, map_location='cuda:0'))
            message = f"model loaded from: {fpath_state_dict}"
            file_it(f'{runpath}log.txt', message, True)

        return model.to(device)
    
    def learn_parameters(self, runpath, loader_0, loader_1, model,
        nr_epochs, lr, lf_weights, weight_decay, train_ap, val_ap,
        device,
    ):
        print('init training ...')
        trainer = MILTrainer(model, lr, lf_weights, weight_decay, train_ap, val_ap, device)
        trainer.execute_train_protocol(
            trainloader=loader_0,
            valloader=loader_1,
            nr_epochs=nr_epochs,
            runpath=runpath,
            device=device,
        )
        return trainer