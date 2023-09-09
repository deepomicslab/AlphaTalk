import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import time
import copy
import torch.optim as optim
from types import SimpleNamespace
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import recall_score

# path for the generated ProtT5 embeddings
EMB_PATH = './data/source/human_v4_ProtT5'


class PairDataset(Dataset):
    def __init__(self, pair_info_df, single_data_pool, reduce_form, emb_path=EMB_PATH):
        self.info = pair_info_df  # [ligand, receptor, ligand_uniprot, receptor_uniprot, y, set]
        self.single_data_pool = single_data_pool
        self.emb_path = emb_path

        if reduce_form is None:
            self.reduce_func = None
        elif reduce_form == 'sum':
            self.reduce_func = torch.sum
        elif reduce_form == 'mean':
            self.reduce_func = torch.mean
        else:
            raise Exception('Unknown reduce form for embedding')

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        row = self.info.iloc[idx, ]
        ligand_file = 'AF-{}-F1-model_v4'.format(row['ligand_uniprot'])
        receptor_file = 'AF-{}-F1-model_v4'.format(row['receptor_uniprot'])
        y = torch.tensor(row['y'], dtype=torch.float32)

        ligand_data = self.single_data_pool[ligand_file]
        receptor_data = self.single_data_pool[receptor_file]

        # emb
        ligand_data.emb = torch.tensor(
            np.load(os.path.join(self.emb_path, f'{ligand_file}.npy')), dtype=torch.float32)
        receptor_data.emb = torch.tensor(
            np.load(os.path.join(self.emb_path, f'{receptor_file}.npy')), dtype=torch.float32)
        if self.reduce_func is not None:
            ligand_data.emb = self.reduce_func(ligand_data.emb, dim=0, keepdim=True)
            receptor_data.emb = self.reduce_func(receptor_data.emb, dim=0, keepdim=True)

        return ligand_data, receptor_data, y


class TrainProcessor:
    def __init__(self, model, loaders, args):
        self.model = model
        self.train_loader, self.val_loader, self.test_loader = loaders
        print('len train_loader.dataset:', len(self.train_loader.dataset))
        print('len val_loader.dataset:', len(self.val_loader.dataset))
        print('len test_loader.dataset:', len(self.test_loader.dataset))
        self.args = args
        self.optimizer, self.scheduler = self.build_optimizer()

    def build_optimizer(self):
        args = self.args
        # return an iterator
        filter_fn = filter(lambda p: p.requires_grad, self.model.parameters())  # params is a generator (kind of iterator)

        # optimizer
        weight_decay = args.weight_decay
        if args.opt == 'adam':
            optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
        elif args.opt == 'sgd':
            optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
        elif args.opt == 'rmsprop':
            optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
        elif args.opt == 'adagrad':
            optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)

        # scheduler
        if args.opt_scheduler == 'none':
            return None, optimizer
        elif args.opt_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
        elif args.opt_scheduler == 'reduceOnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             patience=args.lr_decay_patience,
                                                             factor=args.lr_decay_factor)
        else:
            raise Exception('Unknown optimizer type')

        return optimizer, scheduler

    @torch.no_grad()
    def test(self, model, dataloader):
        model.eval()

        pred_ls = []
        y_ls = []
        for batch_idx, (ligand, receptor, label) in enumerate(dataloader):
            ligand = ligand.to(self.args.device)
            receptor = receptor.to(self.args.device)
            label = label.to(self.args.device)

            pred_ls.append(model(ligand, receptor))
            y_ls.append(label)

        pred = torch.cat(pred_ls, dim=0).reshape(-1)
        y = torch.cat(y_ls, dim=0).reshape(-1)

        # get metrics
        metrics = {}
        metrics['loss'] = model.loss(pred, y).item()
        metrics['acc'] = (pred.round() == y).sum() / len(pred)

        y = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        metrics['auroc'] = roc_auc_score(y, pred)
        metrics['auprc'] = average_precision_score(y, pred)

        pred = np.where(pred > 0.5, 1, 0)
        metrics['recall'] = recall_score(y, pred)

        return SimpleNamespace(**metrics)

    def train(self):

        best_val_loss = 1e16
        best_model = None
        es = 0

        for epoch in range(self.args.epochs):

            epoch_lr = self.optimizer.param_groups[0]['lr']
            train_epoch_loss = 0.0
            self.model.train()

            for batch_idx, (ligand, receptor, label) in enumerate(self.train_loader):
                ligand = ligand.to(self.args.device)
                receptor = receptor.to(self.args.device)
                label = label.to(self.args.device)
                self.optimizer.zero_grad()

                pred = self.model(ligand, receptor)

                loss = self.model.loss(pred, label)
                loss.backward()

                # clip gradients
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)

                self.optimizer.step()
                train_epoch_loss += loss.item()

                # if (batch_idx + 1) % 10 == 0:
                #     print(time.strftime('%H:%M:%S'), 'processed {}/{}'.format(batch_idx + 1, len(train_loader)))

            train_epoch_loss /= len(self.train_loader)

            # ---validation---
            val_metrics = self.test(self.model, self.val_loader)
            val_epoch_loss, val_epoch_acc = val_metrics.loss, val_metrics.acc

            self.model.train()
            if self.args.opt_scheduler is None:
                pass
            elif self.args.opt_scheduler == 'reduceOnPlateau':
                self.scheduler.step(val_epoch_loss)
            elif self.args.opt_scheduler == 'step':
                self.scheduler.step()

            # print training process
            log = 'Epoch: {:03d}/{:03d}; ' \
                  'AVG Training Loss (MSE):{:.5f}; ' \
                  'AVG Val Loss (MSE):{:.5f};' \
                  'AVG Val ACC:{:.5f};' \
                  'lr:{:8f}'
            print(time.strftime('%H:%M:%S'),
                  log.format(
                      epoch + 1,
                      self.args.epochs,
                      train_epoch_loss,
                      val_epoch_loss,
                      val_epoch_acc,
                      epoch_lr
                  ))
            if epoch_lr != self.optimizer.param_groups[0]['lr']:
                print('lr has been updated from {:.8f} to {:.8f}'.format(epoch_lr,
                                                                         self.optimizer.param_groups[0]['lr']))

            # determine whether stop early by val_epoch_loss
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_model = copy.deepcopy(self.model)
                es = 0
            else:
                es += 1
                print("Counter {} of patience {}".format(es, self.args.es_patience))
                if es >= self.args.es_patience:
                    print("Early stopping with best_val_loss {:.8f}".format(best_val_loss))
                    break

        test_metrics = self.test(best_model, self.test_loader)

        return best_model, test_metrics

