import argparse
import os
import time
from datetime import datetime

from tqdm import tqdm 
import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt 
import pickle

from dataset import  MyDataLoader, load_data
import custom_metrics as c_metrics
import model
import sys 
sys.path.append('..')
from utils import Scaler


def do_compute(model, batch, device):
    batch = batch.to(device)
    l_preds, loss, valid_seq_ind = model(batch)
    l_targets = batch.wdfp[:, 1:, 0] # Remove the first time step t = 0
    return  loss, (l_targets, l_preds), valid_seq_ind


def run_batch(model, optimizer, data_loader, epoch_i, desc, device):
        total_loss = 0
        label_list, pred_list, mask_list = [], [], []
        
        for batch in tqdm(data_loader, desc= f'{desc} Epoch {epoch_i}'):
            loss, (l_targets, l_preds),  valid_seq_ind  = do_compute(model, batch, device)
            
            if model.training and isinstance(loss, torch.Tensor):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss)
            
            with torch.no_grad():
                
                l_targets = l_targets.cpu().numpy().T
                assert l_targets.ndim == 2
                
                if not model.training and scaler is not None:
                    l_targets = scaler.inverse_transform(l_targets.reshape(-1, 1)).reshape(l_targets.shape)
                
                l_preds = l_preds.cpu().numpy().T
                assert l_preds.ndim == 2
                if  not model.training and scaler is not None:
                    l_preds = scaler.inverse_transform(l_preds.reshape(-1, 1)).reshape(l_preds.shape)
                
                label_list.append(l_targets)
                pred_list.append(l_preds)

                mask = valid_seq_ind.cpu().numpy().T
                assert np.any(mask)
                mask_list.append(mask)

        
        total_loss /= len(data_loader)

        label_list = np.concatenate(label_list, axis=1) 
        pred_list = np.concatenate(pred_list, axis=1)
        mask_list = np.concatenate(mask_list, axis=1)

        mask_regr = mask_list & np.full_like(pred_list, fill_value=True).astype('bool') 
        mse = c_metrics.MSE_score(label_list, pred_list, mask_regr)
        nse = c_metrics.NSE_score(label_list, pred_list, mask_regr)
        smape = c_metrics.smape_score(label_list, pred_list, mask_regr)
        r = c_metrics.pearson(label_list, pred_list, mask_list)

                    
        return total_loss, mse, nse, smape, r


def print_metrics(loss, mse, nse, smape, r):
    print(f'loss: {loss:.4f}, mse: {mse:.4f}, nse: {nse:.4f}, smape: {smape:.4f}, pearson: {r:.4f}')



def train(train_data_loader, val_data_loader):
    
    for epoch_i in range(1, args.n_epochs+1):
        
        start = time.time()
        model.train()
        train_loss, train_mse, train_nse, train_smape, train_r = run_batch(model, optimizer, train_data_loader, epoch_i,  'train', args.device)
        
        model.eval()
        with torch.no_grad():
            ## Validation 
            if val_data_loader:
                val_loss , val_mse, val_nse, val_smape, val_r = run_batch(model, optimizer, val_data_loader, epoch_i, 'val', args.device)

        if train_data_loader:
            print(f'\n#### Epoch {epoch_i} time {time.time() - start:.4f}s')
            print('#### Training')
            print_metrics(train_loss, train_mse.mean(), train_nse.mean(), train_smape.mean(), train_r.mean())

        if val_data_loader:
            print('#### Validation')
            print_metrics(val_loss, val_mse.mean(), val_nse.mean(), val_smape.mean(), val_r.mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    data_source = '.'
    parser.add_argument('--time_steps', type=int, default=8, help='Total number of time steps.')
    parser.add_argument('--drop', type=float, default=0.3, help='Dropout probability.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learing rate.')
    parser.add_argument('--offset', type=int, default=0, help='Initial time step.')
    parser.add_argument('--v_h_dim', type=int, default=16, help='Vector features - hidden dimensions.')
    parser.add_argument('--s_h_dim', type=int, default=32, help='Scalar features - hidden dimensions.')
    

    args = parser.parse_args()

    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.time_stamp = f'{datetime.now()}'.replace(':', '_')
    args.dataset = 'data'
    
    train_dataset, val_dataset, test_dataset = load_data(args.dataset, args.time_steps, args.offset)
    train_loader = MyDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    valid_loader = MyDataLoader(val_dataset, batch_size=args.batch_size) 
    sample = train_dataset[0]
    args.static_in_dims = sample.s_static.shape[-1]
    s_in_dim = sample.rain.shape[-1] + sample.wdfp.shape[-1] + sample.x_v_norm.shape[-1] + sample.s_static.shape[-1]
    v_in_dim = sample.x_v.shape[-2]
    args.in_dims = (int(s_in_dim), int(v_in_dim))
    
    model = model.FloodGNNGRU(args)
    args.model_name = model.__class__.__name__
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(args.model_name)

    scaler = None
    if True:
        try: 
            with open(f'{args.dataset}/wdfp_scaler.pckl', 'rb') as f:
                scaler = pickle.load(f)
            print(f"{type(scaler)} from {args.dataset}")
        except FileNotFoundError:
            print('Not using any scaler')
    
    model.to(device=args.device)

    print(f'Training on {args.device}.')
    print(f'Starting  at', args.time_stamp)

    print(args)
    print(f'Train on {len(train_dataset)}, Validating on {len(val_dataset)}')
    train(train_loader, valid_loader)
