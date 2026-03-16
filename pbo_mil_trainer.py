
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

from MIL.reader import MILReader
from MIL.util import file_it, plot_performance
from pbo_mil_model import init_model

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

    def log_epoch(self, header, dpath_run, nr_epochs):
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
            fpath_model = dpath_run / "model.pth"
            record_history(fpath_model)
            torch.save(self.model.state_dict(), fpath_model)
            epoch_info = f'{epoch_info} saved!'
            
        if len(self.valcost) <= 1:
            file_it(dpath_run / 'log.txt', '\n' + header, False)
        file_it(dpath_run/ "log.txt", epoch_info, True)
        
    def execute_train_protocol(self, trainloader, valloader, nr_epochs, dpath_run, device):
        print(f'\nbeginning training {str(datetime.now())[11:19]}')
        header = f'epoch\tloss\t\taccuracy\tpred_pos\ttrue_pos\tprecision\trecall  \ttime'
        print(header)

        for _ in range(1, nr_epochs + 1):
            self.train_epoch(trainloader, device)
            self.val_epoch(valloader, device)
            self.log_epoch(header, dpath_run, nr_epochs)

def record_history(fpath_model):
    if not fpath_model.is_file():
        with open(f'run/history/history.txt', 'a') as file:
            file.write(f"\n{fpath_model}")

class MilTrainWrapper:
    def __init__(self, dpath_ptFeature, dpath_milfolds,
        testfold_name, hparam, fpath_state_dict, tag, device,
    ):
        fpaths_milfolds = list(dpath_milfolds.iterdir())
        fpaths_trainfolds = [
            fpath for fpath in fpaths_milfolds
            if not fpath.name == testfold_name
        ]

        for k, fpath_valfold in enumerate(fpaths_trainfolds):
            map_val = pd.read_csv(fpath_valfold).sample(frac=1)
            map_0 = None
            for fpath_trainfold in fpaths_trainfolds:
                if fpath_trainfold == fpath_valfold:
                    continue

                map = pd.read_csv(fpath_trainfold)
                if map_0 is None:
                    map_0 = map
                else:
                    map_0 = pd.concat([map_0, map], axis=0)
            map_0 = map_0.sample(frac=1).reset_index(drop=True)

            map0_ids = map_0['slide_id'].to_list()
            assert not any([id in map0_ids for id in map_val['slide_id'].to_list()])

            loader_0, loader_1 = self.init_loaders(
                dpath_features_pt=dpath_ptFeature,
                map_0=map_0,
                map_val=map_val,
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
            
            dpath_run = self.init_run(hparam, logmore, tag, device, k)
            model = init_model(dpath_run, fpath_state_dict, hparam['dropout'], device)
            self.trainer = self.learn_parameters(
                dpath_run=dpath_run,
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

            plot_performance(self.trainer, dpath_run)

    def init_loaders(self, dpath_features_pt, map_0, map_val, batch_size):
        reader_0 = MILReader(map_0, dpath_features_pt, 'train')
        reader_1 = MILReader(map_val, dpath_features_pt, 'train')
        loader_0 = DataLoader(reader_0, batch_size, shuffle=True)
        loader_1 = DataLoader(reader_1, batch_size, shuffle=True)
        return loader_0, loader_1

    def init_run(self, hparam, logmore, tag, device, k):
        current = datetime.now()
        # create directory for output files of current run
        dpath_run = Path(f'run/{tag}/{k}|_{str(current)[8:10]}_{str(current)[11:13]}_{str(current)[14:16]}_{str(current)[17:19]}')
        dpath_run.mkdir(exist_ok=True, parents=True)
        # create dir for run history if none exists
        Path('run/history/').mkdir(exist_ok=True, parents=True)
        # saving runlog to current run folder
        fpath_log = dpath_run / 'log.txt'
        with open(fpath_log, 'a') as file:
            file.write('################# initiating run #################\n')
            file.write(f'run dir has been created in {dpath_run}\n\n')
            for key, val in {**hparam, **logmore}.items():
                file.write(f'{key}\t: {val}\n')
            file.write('\n')
            file.write(f'Using {device} device\n')
            file.write('################# initiated run ##################\n')
        # printing current state of logfile to terminal
        with open(fpath_log, 'r') as file:
            for_terminal = file.read()
        print(for_terminal)
        return dpath_run

    """
    def init_model(self, dpath_run, fpath_state_dict, dropout, device):
        print('initiating model ...')
        from pbo_mil_model import CLAM_SB
        model = CLAM_SB(dropout)

        if not fpath_state_dict == '':
            model.load_state_dict(torch.load(fpath_state_dict, map_location='cuda:0'))
            message = f"model loaded from: {fpath_state_dict}"
            file_it(dpath_run / 'log.txt', message, True)

        return model.to(device)
    """
    
    def learn_parameters(self, dpath_run, loader_0, loader_1, model,
        nr_epochs, lr, lf_weights, weight_decay, train_ap, val_ap,
        device,
    ):
        print('init training ...')
        trainer = MILTrainer(model, lr, lf_weights, weight_decay, train_ap, val_ap, device)
        trainer.execute_train_protocol(
            trainloader=loader_0,
            valloader=loader_1,
            nr_epochs=nr_epochs,
            dpath_run=dpath_run,
            device=device,
        )
        return trainer