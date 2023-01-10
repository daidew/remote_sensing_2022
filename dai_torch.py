import time
import warnings
import numpy as np
np.set_printoptions(suppress=True)
import torch
import rasterio
import cv2
import cv2 as cv
import os
import os.path as osp
import joblib
import argparse
import open_earth_map.oem as oem

from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import scipy.stats

import torchvision
from torchvision import transforms
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


import torch_geometric.transforms as T
import torch_geometric
from torch_geometric.nn import GCNConv, GeneralConv, GINConv, GATv2Conv
from torch_geometric.logging import init_wandb, log
from torch_geometric.data import InMemoryDataset, Data

from typing import Callable, List, Optional


class SegGraphDataset(InMemoryDataset):
    """
    Training, validation and test splits are given by binary masks.
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root: str, transform: Optional[Callable] = None,pre_transform: Optional[Callable] = None, load_etc=False, custom_fns=None, custom_data=None):
                
        super().__init__(root, transform, pre_transform)
        # bug with filename_indices; seem to have overwritten for the val set.. (the val is included in train as well?)
        with open(osp.join(self.filename_indices_dir(), 'filename_indices.pkl'), 'rb') as f2:
            self.idx2fn, self.fn2idx = joblib.load(f2)
        
        self.transform = transform
        self.pre_transform = pre_transform
         
        self.is_included = None
        if custom_fns is not None:
            self.is_included = []
            for i in range(len(self.idx2fn)):
                if os.sep.join(self.idx2fn[i].split('-')) in custom_fns:
                    self.is_included.append(i)
            print('included data:', len(self.is_included))
            
#         if custom_data is not None:
#             print('pre-specified custom data')
#             self.data, self.slices = custom_data
#         else:
#             print(f'loading data, slices from {self.processed_paths[0]}')
#             self.data, self.slices = torch.load(self.processed_paths[0])
            
        self.etc_dir = osp.join(root, 'etc')
        self.etc = []
        if load_etc:
            print(f'loading {len(self.is_included)} etc files..')
            for i in tqdm(range(len(self.is_included))):
                self.etc.append(joblib.load(osp.join(self.etc_dir, 'etc.'+self.idx2fn[self.is_included[i]])))
        
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')
    
    def filename_indices_dir(self) -> str:
        return osp.join(self.root, 'filename_indices')
    
    @property
    def raw_file_names(self) -> List[str]:     
        raw_graph_fns = [fn for fn in os.listdir(self.raw_dir) if fn[:3] == 'ind']
        raw_label_fns = [fn for fn in os.listdir(self.raw_dir) if fn[:5] == 'label']
        raw_feat_fns  = [fn for fn in os.listdir(self.raw_dir) if fn[:4] == 'feat'] 
        return raw_graph_fns, raw_label_fns, raw_feat_fns

    @property
    def processed_file_names(self) -> str:
#         return ['data_'+fn[4:]+'.pt' for fn in os.listdir(self.raw_dir) if fn[:3] == 'ind']
        return 'data_all.pt'

    def read_graph_from_filepath(self, fp):
        arr = None
        edge_index = []
        with open(fp, 'r') as f2:
            arr = [[int(v) for v in st.strip().split(';')] for st in f2.readlines()]
        for i in range(len(arr)):
            v = arr[i][0]
            for j in range(1,len(arr[i])):
                edge_index.append([v, arr[i][j]])
        return torch.tensor(edge_index).T 
        
    def read_data(self):
        raw_graph_fns, raw_label_fns, raw_feat_fns = self.raw_file_names
        assert len(raw_graph_fns) == len(raw_label_fns), "len(raw_graph_fns) not equal to len(raw_label_fns)."
        d = {}
        for raw_label_fn in raw_label_fns:
            # load labels
            fn = raw_label_fn[6:]
            d[fn] = [torch.tensor(np.load(osp.join(self.raw_dir, raw_label_fn)), dtype=torch.long)]
            
        # load graphs
        for raw_graph_fn in raw_graph_fns:
            fn = raw_graph_fn[4:]
            edge_index = self.read_graph_from_filepath(osp.join(self.raw_dir, raw_graph_fn))
            d[fn].append(edge_index)
        
        # load feats
        for raw_feat_fn in raw_feat_fns:
            fn = raw_feat_fn[5:]
            x = torch.tensor(np.load(osp.join(self.raw_dir, raw_feat_fn)), dtype=torch.float)
            d[fn].append(x)
        
        data_lst = []
        self.idx2fn = []
        self.fn2idx = {}
        for idx, fn in enumerate(sorted(d.keys())):
            data = Data(x=d[fn][2], edge_index=d[fn][1], y=d[fn][0])
            data_lst.append(data)
            self.idx2fn.append(fn)
            self.fn2idx[fn] = idx
            torch.save(data, osp.join(self.processed_dir, 'data_'+fn+'.pt'))
        
        with open(osp.join(self.filename_indices_dir(), 'filename_indices.pkl'), 'wb') as f2:
            joblib.dump((self.idx2fn, self.fn2idx), f2)
        
        return data_lst
        
    def process(self):
        data_lst = self.read_data()
        
        if self.pre_filter is not None:
            data_lst = [data for data in data_lst if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_lst = [self.pre_transform(data) for data in data_lst]
        
        data, slices = self.collate(data_lst)
        torch.save((data, slices),self.processed_paths[0])
        
    def __getitem__(self, idx):
        if self.is_included is not None:
            assert idx < len(self.is_included)
            idx = self.is_included[idx]

        data = torch.load(osp.join(self.processed_dir, f'data_{self.idx2fn[idx]}.pt'))
            
        if self.transform is not None:
            return self.transform(data)
        return data
    
    def __len__(self):
        if self.is_included: 
            return len(self.is_included)
        else:
            return len(self.idx2fn)
    
    def __repr__(self) -> str:
        return f'{self.name}()'

class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1024, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 9)
        
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    
    
class GeneralGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GeneralConv(1024, 128, aggr='sum', attention=True, l2_normalize=True)
        self.conv_lst = nn.ModuleList([GeneralConv(128, 128, aggr='sum', attention=True, l2_normalize=True) for _ in range(1)])
        self.conv3 = GeneralConv(128, 9, aggr='sum', attention=True, l2_normalize=True)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        for gc in self.conv_lst:
            x = gc(x, edge_index)
            x = F.relu(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

class GATv2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc_lst = nn.ModuleList([nn.Linear(1024, 512), nn.Linear(512, 256), nn.Linear(256, 64)])
        
        self.conv_lst = nn.ModuleList([GATv2Conv(64, 64, heads=7, concat=False) for _ in range(5)])
        
        self.fc_lst2 = nn.ModuleList([nn.Linear(64, 32), nn.Linear(32, 16), nn.Linear(16, 9)])

        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for fc in self.fc_lst:
            x = F.relu(fc(x))
        
        for gc in self.conv_lst:
            x = F.relu(gc(x, edge_index))
#             x = torch.nn.MaxPool1d(1, stride=1)(x)

        for fc in self.fc_lst2:
            x = fc(F.relu(x))
            
        return F.log_softmax(x, dim=1)

class CroppedImageDataset(Dataset):
    def __init__(self, fn, img, partition_res, res_lst, num_nodes, transform=None, target_transform=None):
        self.partition_res = partition_res
        self.res_lst = res_lst
        self.fn = fn
        self.img = img
        self.num_nodes = num_nodes
        self.transform = transform
        self.target_transform = target_transform
        self.minmaxcoords = []
        for i in range(len(res_lst)):
            t = np.min(np.array(self.res_lst[i], dtype=int), axis=0)
            minx, miny = t[0],t[1] 
            t = np.max(np.array(self.res_lst[i], dtype=int), axis=0)
            maxx, maxy = t[0],t[1] 
            self.minmaxcoords.append((minx,miny,maxx,maxy))
        
    def __len__(self):
        return self.num_nodes

    def __getitem__(self, node_idx):
        assert node_idx < self.num_nodes, f'node_idx: {node_idx} > num_nodes={self.num_nodes}'
        minx, miny, maxx, maxy = self.minmaxcoords[node_idx]
        cropped_img = self.img[minx:maxx+1, miny:maxy+1]
        for t in self.res_lst[node_idx]:
            cropped_img[t[0]-minx, t[1]-miny, :] = 0
        # plt.imshow(cropped_img_lst[0][0].transpose(0, -1)) (plot image)
#         cropped_img_lst.append(preprocess(cropped_img).float().unsqueeze(0))
        if self.transform:
            cropped_img = self.transform(cropped_img)
        return cropped_img

    
def dice_loss(y_maps_small, target, device, num_classes=9, eps=1e-5):
    p = F.softmax(y_maps_small, dim=-1)
    g = F.one_hot(target, num_classes=num_classes)
    eps = torch.tensor(eps).to(device)
    pg = (p * g).sum((0, 1, 2)) + eps
    p2q2 = (p*p).sum((0, 1, 2)) + (g).sum((0, 1, 2)) + eps
    d = 2 * pg / p2q2
    return d