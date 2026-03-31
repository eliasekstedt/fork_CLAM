
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from MIL.reader import MILReader
from MIL.util import file_it
from MIL.model import init_model


class Evaluator:
    def __init__(self, dpath_milfolds, dpath_ptFeature,
            testfold_name, hparam, tag, device
    ):
        dpath_tag = Path(f'run/{tag}')
        fpaths_models = []
        [
            [fpaths_models.append(path) for path in lst if path.name == 'model.pth']
            for lst in [
                dir.iterdir()
                for dir in list(dpath_tag.iterdir())
                if dir.is_dir()
            ]
        ]

        fpath_testfold = dpath_milfolds / testfold_name
        df_test = pd.read_csv(fpath_testfold)
        loader = DataLoader(
            MILReader(df_test, dpath_ptFeature, 'test'),
            batch_size=hparam['batch_size'],
            shuffle=False,
        )

        df_eval = self.evaluate(dpath_tag, fpaths_models, hparam['dropout'], loader, device)
        self.get_stats(dpath_tag, df_eval, hparam['dropout'])

    def evaluate(self, dpath_tag, fpaths_models, dropout, testloader, device):
        df_eval = []
        for fpath_model in fpaths_models:
            model = init_model(dpath_tag, fpath_model, dropout, device)
            model.eval()
            for bag_id, features, label in testloader:
                with torch.inference_mode():
                    features, label = features.squeeze(0).to(device), label.to(device)
                    logit, _, _, _ = model(features, label=label, instance_eval=True)

                    pred_item = logit.argmax(1).item()
                    label_item = label.item()
                    df_eval.append({
                        'model':fpath_model.parent.name,
                        'bag_id':bag_id[0],
                        'logit_0':logit.tolist()[0][0],
                        'logit_1':logit.tolist()[0][1],
                        'label':label_item,
                        'pred':pred_item,
                    })

        df_eval = pd.DataFrame(df_eval)
        df_eval.to_csv(dpath_tag / 'df_eval.csv', index=False)
        return df_eval

    def get_stats(self, dpath_tag, df_eval, dropout):
        pred_vote = df_eval.groupby(by=['bag_id'])['pred'].mean()
        label_vote = df_eval.groupby(by=['bag_id'])['label'].mean()
        stats = pd.DataFrame(pred_vote).merge(label_vote.astype(int), on='bag_id').reset_index(drop=False)
        stats['final_pred'] = (stats['pred'] > 0.5).astype(int)

        n_TP = stats[(stats['label'] == 1) & (stats['final_pred'] == 1)].shape[0]
        n_TN = stats[(stats['label'] == 0) & (stats['final_pred'] == 0)].shape[0]
        n_FN = stats[(stats['label'] == 1) & (stats['final_pred'] == 0)].shape[0]
        n_FP = stats[(stats['label'] == 0) & (stats['final_pred'] == 1)].shape[0]
        tot = stats.shape[0]
        message = 'dropout: {}| TP: {}, TN: {}, FN: {}, FP: {}, tot: {}'.format(
            dropout, n_TP, n_TN, n_FN, n_FP, tot
        )

        fpath_stats = Path('run/stats.txt')
        file_it(file_name=fpath_stats, message=message, to_terminal=True)
        stats.to_csv(dpath_tag / 'stats.csv', index=False)
    
    
