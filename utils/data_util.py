import os
import _pickle as cPickle
import torch
import numpy as np
import sys; sys.path.append('.')
from torch.utils.data import Dataset, DataLoader
from .macro import PAD_IDX

class CADSequenceDataset(Dataset):
    def __init__(self, data_dir, mode, max_seq_len=None):
        path = os.path.join(data_dir, mode) + '.pkl'
        with open(path, 'rb') as f:
            obj = cPickle.load(f)
        self.data = list(filter(lambda x: x is not None, obj['sequences']))  # list of numpy arrays
        self.max_seq_len = max(map(np.shape, self.data)) if max_seq_len is None else max_seq_len
    def pad_max_seq_len(self, seq: np.ndarray):
        curr_seq_len = seq.shape[0]
        padded_seq = torch.ones(self.max_seq_len, dtype=int) * PAD_IDX
        padded_seq[:curr_seq_len] = torch.LongTensor(seq)
        return padded_seq
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.pad_max_seq_len(self.data[idx])

def get_dataloader(data_dir, mode, batch_size):
    dataset = CADSequenceDataset(data_dir, mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if mode == 'train' else False
    )

if __name__ == '__main__':
    dataset = CADSequenceDataset('../datasets/cad_data/cad_seq/BRepCheck/', 'test')
    print(dataset[1])
    print(dataset[1].shape)
