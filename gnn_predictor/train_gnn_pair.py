import os
import random
import torch
import numpy as np
import pandas as pd
from models import GNNPair
from types import SimpleNamespace
from utils import PairDataset, TrainProcessor
from torch_geometric.loader import DataLoader

random.seed(42); np.random.seed(42)


if __name__ == '__main__':
    # args
    args = {
        'batch_size': 52,
        'model_reduce': 'sum',
        'emb_reduce': 'sum',
        'epochs': 200,
        'device': 'cuda',
        'opt': 'adam',
        'opt_scheduler': 'step',
        'opt_decay_step': 5,
        'opt_decay_rate': 0.9,
        'weight_decay': 1e-3,
        'lr': 3e-4,
        'es_patience': 10,
        'save': True,
        'pt_data': './data/graph/human_v4_6_alpha.pt'
    }
    args = SimpleNamespace(**args)

    # data related
    df_dataset = pd.read_csv('./data/df/df_human_dataset_nfold1.csv')
    df_train = df_dataset.query('set == "train"').reset_index(drop=True)
    df_val = df_dataset.query('set == "val"').reset_index(drop=True)
    df_test = df_dataset.query('set == "test"').reset_index(drop=True)

    single_data_pool = {data['name']: data for data in torch.load(args.pt_data)}

    # get dataloader
    train_data = PairDataset(df_train, single_data_pool=single_data_pool, reduce_form=args.emb_reduce)
    val_data = PairDataset(df_val, single_data_pool=single_data_pool, reduce_form=args.emb_reduce)
    test_data = PairDataset(df_test, single_data_pool=single_data_pool, reduce_form=args.emb_reduce)

    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size)

    print(f"Start GNNPair with args: \n", args.__dict__)

    for i in range(5):
        model = GNNPair(input_dim=128, hidden_dim=128, num_layers=1, reduce_form=args.reduce_form)
        model.to(args.device)
        print(model)
        train_val = TrainProcessor(
            model=model,
            loaders=[train_data_loader, val_data_loader, test_data_loader],
            args=args
        )
        best_model, test_metrics = train_val.train()
        print('test loss: {:5f}; test acc: {:4f}'.format(test_metrics.loss, test_metrics.acc))

        if args.save:
            save_dir = './result/{}_{}'.format(os.path.basename(args.pt_data).replace('.pt', ''), 'GNNPair')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_path = os.path.join(save_dir,
                                     '{}_acc{:.3f}_roc{:.3f}_prc{:.3f}_recall{:.3f}.pt'.format(i,
                                                                                               test_metrics.acc,
                                                                                               test_metrics.auroc,
                                                                                               test_metrics.auprc,
                                                                                               test_metrics.recall))
            torch.save(best_model.state_dict(), save_path)
