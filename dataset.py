import torch 
import wandb 
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset, DataLoader 
from torch_geometric.data import Data, Batch
from tqdm import tqdm 


def get_sliding_chunks(data, T, dataset_names, window_size=6):
    print(f'[Sliding window] Working with {len(dataset_names)} datasets: {dataset_names}')
    
    offset = 0
    data_collection_list = []
    for n in dataset_names:
        dataset = data[n]
        for d in dataset:
            data_d = d["data"][:, offset:][:, :T+1] 
            bin_d = d["bin"][:, offset:][:, :T+1] 
            seq_len = data_d.shape[1]
            true_seq = d["seq"][offset:][:T+1] 
            true_seq[-1] = False
            
            if np.all(~true_seq): continue
            
            temp_list = []
            for c in range(seq_len):
                temp_true_seq = true_seq[c:c + window_size]
                assert temp_true_seq.ndim == 1
                
                if np.all(~temp_true_seq): break # If all the remaining sequences are nill
                
                temp_d = data_d[:, c:c + window_size]
                temp_bin_d = bin_d[:, c:c + window_size]
                            
                if temp_d.shape[1] < window_size:
                    padded_shape = list(temp_d.shape)
                    padded_shape[1] = window_size
                    padded_d = np.zeros(padded_shape)
                    padded_bin_d = np.zeros((*padded_shape[:2], *temp_bin_d.shape[2:])).astype('int')
                    padded_d[:, :temp_d.shape[1]] = temp_d
                    padded_bin_d[:, :temp_bin_d.shape[1]] = temp_bin_d
                    padded_seq = np.full(padded_shape[1], fill_value=False)
                    padded_seq[:temp_d.shape[1]] = temp_true_seq
                    temp_d = padded_d
                    temp_bin_d = padded_bin_d
                    temp_true_seq = padded_seq
                
                temp_list.append({
                    'static': d['static'],
                    'start': d['start'],
                    'data': temp_d,
                    'seq': temp_true_seq[:-1],
                    'bin': temp_bin_d,
                    'edges': d['s_edges'],
                })
                if len(d['static']) > 2:
                    assert d['s_edges'].max() > 1
            
            if len(temp_list): data_collection_list.extend(temp_list)
    
    assert data_collection_list[0]['data'].shape[1] == window_size

    return data_collection_list


def get_data(data, T, dataset_names, offset):
    print(f'Working with {len(dataset_names)} datasets: {dataset_names}')
    
    offset = 0
    data_collection_list = []
    for n in dataset_names:
        dataset = data[n]
        for d in dataset:
            data_d = d["data"][:, offset:][:, :T+1] 
            bin_d = d["bin"][:, offset:][:, :T+1] 
            seq_len = data_d.shape[1]
            true_seq = d["seq"][offset:][:T]
                
            if seq_len <= T: # +1 because we always predict the next time step
                padded_shape = list(data_d.shape)
                padded_shape[1] = T + 1
                padded_d = np.zeros(padded_shape)
                padded_bin_d = np.zeros((*padded_shape[:2], *bin_d.shape[2:])).astype('int')
                padded_d[:, :data_d.shape[1]] = data_d
                padded_bin_d[:, :bin_d.shape[1]] = bin_d
                padded_seq = np.full(T, fill_value=False)
                padded_seq[:data_d.shape[1]] = true_seq
                padded_seq[:, (seq_len-1):] = False # The last entry is always False, because there is output for it
                data_d = padded_d
                bin_d = padded_bin_d
                true_seq = padded_seq

            data_collection_list.append({
                'static': d['static'],
                'start': d['start'],
                'data': data_d,
                'seq': true_seq,
                'bin': bin_d,
                'edges': d['s_edges'],
            })
            
            if len(d['static']) > 2:
                assert d['s_edges'].max() > 1

    assert data_collection_list[0]['data'].shape[1] == (T+1)

    return data_collection_list


def load_data(path, T, offset):
    
    dataset_names = ["harvey"]
    print('Loading training set...')
    train_data = np.load(f"{path}/train.npz", allow_pickle=True)
    print('Loading val set...')
    val_data = np.load(f"{path}/val.npz", allow_pickle=True)
    print('Loading test set')
    test_data = np.load(f"{path}/test.npz", allow_pickle=True)


    train_data = get_sliding_chunks(train_data, T, dataset_names)
    val_data = get_data(val_data, T, dataset_names, offset)
    test_data = get_data(test_data, T, dataset_names, offset)
    
    print('Training dataset...')
    train_dataset = MyDataset(train_data)
    print('Validation dataset...')
    val_dataset = MyDataset(val_data)
    print('Test dataset...')
    test_dataset = MyDataset(test_data)

    return train_dataset, val_dataset, test_dataset


class MyDataset(Dataset):

    def __init__(self, data):
        self.data = self.build_graph_data(data)
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    def collate(self, batch):
        return Batch.from_data_list(batch, follow_batch=['s_static'])
    
    def build_graph_data(self, data):
        data_list = []
        
        for d in tqdm(data, desc='Building data'):
            rain = d['data'][..., -1:]
            wdfp = d['data'][..., :1]
            x_v = np.stack([d['data'][..., 1:3], d['data'][..., 3:5]], axis=-2)
            x_v_norm = d['data'][..., 5:-1]
            temp_seq = np.full((wdfp.shape[0], d['seq'].shape[0]), fill_value=False)
            temp_seq[:,] = d['seq']
            data_list.append(Data(edge_index= torch.LongTensor(d['edges'].T),
                                    s_static=torch.FloatTensor(d['static']), 
                                    seq=torch.BoolTensor(temp_seq), 
                                    x_v=torch.FloatTensor(x_v), 
                                    rain=torch.FloatTensor(rain),
                                    wdfp=torch.FloatTensor(wdfp),
                                    x_v_norm=torch.FloatTensor(x_v_norm),
                                    bin=torch.FloatTensor(d['bin']),
                                    num_nodes=wdfp.shape[0]))
        return data_list

class MyDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=dataset.collate, **kwargs)



