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
from torch_geometric.nn import GCNConv, GeneralConv
from torch_geometric.logging import init_wandb, log
from torch_geometric.data import InMemoryDataset, Data

from typing import Callable, List, Optional

from dai_torch import *

def draw(B, x, y, zx, zy):
    while x < zx and y < zy:
        B[x, y] = True
        x += 1
        y += 1
    while x < zx:
        B[x, y] = True
        x += 1
    while y < zy:
        B[x, y] = True
        y += 1

def connect_pt(B1, s, B2 = None):
    n, m = B1.shape
    B2 = np.zeros((n, m), dtype=bool)
    x = s
    while x < n-s:
        y = s
        while y < m-s:
            if B1[x, y]:
                maxz, conx, cony = -1, False, False
                for zx in range(x, x+s):
                    for zy in range(y, y+s):
                        if B1[zx, zy]:
                            draw(B2,x,y,zx,zy)
                            
            if x >= n-s or y >= m-s:
                break
            y += 1
        x += 1
    return B2

def bfs_xy(edges, 
           img,
           t, 
           thresh=0.05,
           cnt_area_max=20, 
           B = None,
           Qx = None, 
           Qy = None):
    Qx[0], Qy[0] = t
    c = img[Qx[0], Qy[0], :]
    res = []
    if(edges[t]):
        B[t] = True
        return res, c
    n, m = edges.shape
    s = 1
    cnt = 1
    while s > 0:
        s -= 1
        x, y = Qx[s], Qy[s]
        try:
            if np.sum((img[x, y, :] - c)**2)**0.5 > thresh and cnt >= cnt_area_max:
                continue
        except:
            pass
        res.append((x, y))
        c = (c*cnt + img[x, y, :])/(cnt+1)
        cnt += 1
        B[x, y] = True
        if x + 1 < n:
            if B[x+1, y] == 0 and edges[x+1, y] == 0:
                Qx[s], Qy[s] = (x+1, y)
                s += 1
        if x - 1 >= 0:
            if B[x-1, y] == 0 and edges[x-1, y] == 0:
                Qx[s], Qy[s] = (x-1, y)
                s += 1
        if y + 1 < m:
            if B[x, y+1] == 0 and edges[x, y+1] == 0:
                Qx[s], Qy[s] = (x, y+1)
                s += 1
        if y - 1 >= 0:
            if B[x, y-1] == 0 and edges[x, y-1] == 0:
                Qx[s], Qy[s] = (x, y-1)
                s += 1
    return res, c

def count_cluster(dst, B=None, cnt=None):
    n, m = dst.shape
    cnt = 0
    B = np.array(dst)
    for i in range(n):
        for j in range(m):
            if not B[i, j]:
                B |= bfs_xy(B, (i, j))
                cnt += 1
    return cnt

def partition_image(edges, img, thresh=0.01, cnt_area_max=100):
    n, m = edges.shape
    partition_res = np.zeros(edges.shape, dtype=int)
    mean_c_res, cnt_res_lst, res_lst = [np.zeros((3,))], [0], [[]]
    check = np.zeros(edges.shape, dtype=bool)
    B = np.zeros(edges.shape, dtype=bool)
    Qx = np.zeros((np.product(edges.shape),), dtype=int)
    Qy = np.zeros((np.product(edges.shape),), dtype=int)
    num_nodes = 0
    for i in range(n):
        for j in range(m):
            if check[i, j]:
                continue
            res, c = bfs_xy(edges, img, (i, j), thresh=thresh, cnt_area_max=cnt_area_max, B=B, Qx=Qx, Qy=Qy)
            
            if len(res) == 0: # it is a border/edge
                check[i, j] = True
            else:
                num_nodes += 1
                for t in res:
                    partition_res[t] = num_nodes
                    check[t] = True
                mean_c_res.append(c)
                cnt_res_lst.append(len(res))
                res_lst.append(res)
                
    mean_c_res = np.stack(mean_c_res)
    eliminate_border(img, partition_res, mean_c_res, res_lst, num_nodes)
    partition_res, mean_c_res, res_lst, cnt_res_lst = adjust_index_to_start_from_zero(partition_res, mean_c_res, res_lst, cnt_res_lst)
    #eliminate_noise()
    
    return partition_res, mean_c_res, res_lst, cnt_res_lst, num_nodes

def generate_colormap_by_node(partition_res, num_nodes):
    colors = np.random.randint(0, 256, size=(num_nodes+1, 3))
    colors[0] = np.zeros((1, 3))
#     colored_res = np.zeros(partition_res.shape+(3,), dtype=int)
#     for i in range(partition_res.shape[0]):
#         for j in range(partition_res.shape[1]):
#             colored_res[i, j] = colors[partition_res[i, j]]
    colors = np.random.randint(0, 256, size=(num_nodes+1, 3))
    colors[0] = np.zeros((1, 3))
    colored_res = colors[partition_res]
    
    return colored_res

def generate_colormap_by_mean_c(partition_res, mean_c_res):
    colored_res = np.zeros(partition_res.shape+(3,), dtype=float)
    for i in range(partition_res.shape[0]):
        for j in range(partition_res.shape[1]):
            if partition_res[i, j] != 0:
                colored_res[i, j] = mean_c_res[partition_res[i, j]]
    return colored_res

def eliminate_noise(partition_res):
    n, m = partition_res.shape
    for i in range(n):
        for j in range(m):
            d = [(i-1,j-1), (i-1, j), (i, j-1), (i, j+1), (i+1, j), (i+1, j-1), (i-1, j+1), (i+1, j+1)]
            cnt = set()
            for p, q in d:
                if 0 <= p < n and 0 <= q < m:
                    cnt.add(partition_res[p, q])
            res = list(cnt)
            if len(res) == 1 and res[0] != partition_res[i, j]:
                partition_res[i, j] = res[0]

def eliminate_border(img, partition_res, mean_c_res, res_lst, num_nodes):
    # partition_res has not been reindexed yet! 0 represents border/edge that needs to be removed
    n, m = partition_res.shape
    for i in range(n):
        for j in range(m):
            if partition_res[i, j] == 0: # is a border/edge
                minc, bp, bq = 10000, -1, -1
                d = [(i-1,j-1), (i-1, j), (i, j-1), (i, j+1), (i+1, j), (i+1, j-1), (i-1, j+1), (i+1, j+1)]
                for p,q in d:
                    if 0 <= p < n and 0 <= q < m:
                        if partition_res[p, q] != 0: # compare color distance with adjacent non-border/edge cells. 
                            dist = np.sum( (mean_c_res[partition_res[p, q]] - img[i, j])**2)**0.5
                            if dist < minc:
                                minc = dist
                                bp, bq = p,q
                if bp != -1:
                    partition_res[i, j] = partition_res[bp, bq]
                    res_lst[partition_res[bp, bq]].append((i, j))
                else:
                    z = np.random.randint(1, num_nodes+1)
                    partition_res[i, j] = z
                    res_lst[partition_res[bp,bq]].append((i, j))

def generate_graph_from_partition(partition_res):
    n, m = partition_res.shape
    E ,V = {}, set()
    for i in range(n):
        for j in range(m):
            d = [(i-1,j-1), (i-1, j), (i, j-1), (i, j+1), (i+1, j), (i+1, j-1), (i-1, j+1), (i+1, j+1)]
            for p,q in d:
                if 0 <= p < n and 0 <= q < m:
                    if partition_res[p, q] != partition_res[i, j]:
                        V |= {partition_res[p, q], partition_res[i, j]}
                        if partition_res[p, q] in E:
                            E[partition_res[p, q]] |= {partition_res[i, j]}
                        else:
                            E[partition_res[p, q]] = {partition_res[i, j]}
                        if partition_res[i, j] in E:
                            E[partition_res[i, j]] |= {partition_res[p, q]}
                        else:
                            E[partition_res[i, j]] = {partition_res[p, q]}
    return (V, E)                

def score(node_idx, partition_res, res_lst, msk_discrete):
    # r score: freq. of mode of classes in the segmented region (indexed by node idx.), / num. pixels of the segmented region 
    assert node_idx >= 0
    d = np.array([msk_discrete[t] for t in res_lst[node_idx]], dtype=int)
#     print(d)
    p_v = d.shape[0]
    if(p_v == 0):
        print('WARNING: p_v == 0')
        return 0, 0
    else:
        s = scipy.stats.mode(d).count[0] / p_v
        return s, p_v

def avg_acc(num_nodes, partition_res, res_lst, msk_discrete):    
    # avg. r score (avg. class contamination --> ideally, we want r to be as close to 1 as possible)
    s_lst, p_lst = [], []
    for node_idx in range(num_nodes):
        s_i, p_i = score(node_idx, partition_res, res_lst, msk_discrete)
        s_lst.append(s_i)
        p_lst.append(p_i)
    s_lst = np.array(s_lst)
    p_lst = np.array(p_lst)
    s = np.sum(p_lst)
    if s == 0:
        print('WARNING: p_lst is empty')
        return 0, [], []
    else:
        r = s_lst @ p_lst / s
        return r, s_lst, p_lst

def meanIOU(msk_discrete, pred_discrete):
    ious = []
    for i in range(9):
        itsec = np.sum((pred_discrete == i) & (msk_discrete == i))
        union = np.sum((pred_discrete == i) | (msk_discrete == i))
        if union == 0:
            itsec, union = -1, 1
        else:
            ious.append(itsec/union)
#         print(f'IOU ({i}): {itsec/union}')

#     print('mean IOU:', np.mean(ious), 'num. counted classes:', len(ious))
    miou = np.mean(ious)
    return miou
    
def generate_labels_from_graph(res_lst, msk_discrete, num_nodes):
    y_lst = []
    for node_idx in range(num_nodes): #NODE INDEX START WITH 0
        d = np.array([msk_discrete[t] for t in res_lst[node_idx]], dtype=int)
        y_lst.append(scipy.stats.mode(d).mode[0])
    return np.array(y_lst)

def adjust_index_to_start_from_zero(partition_res, mean_c_res, res_lst, cnt_res_lst):
    assert np.all(partition_res != 0), "zero entries exists in partition_res."
    assert cnt_res_lst[0] == 0, f"the first entry is non-empty: cnt_res_lst[0]={cnt_res_lst[0]}."
    mean_c_res = mean_c_res[1:]
    res_lst = res_lst[1:]
    cnt_res_lst = cnt_res_lst[1:]
    partition_res = partition_res - 1
    return partition_res, mean_c_res, res_lst, cnt_res_lst

def eval_one_graph(model, idx, dataset, device, plot=False, log_image=False):
    assert idx < len(dataset), f'idx={idx} < len(dataset) = {len(dataset)}'
    with torch.no_grad():
        out = model(dataset[idx].to(device))
        out = out.detach().cpu().numpy()
        partition_res_small, res_lst_small, msk_discrete = [dataset.etc[idx][st] for st in ['partition_res_small', 'res_lst_small', 'msk_discrete']]
        y_pred_small = np.argmax(out, axis=1)
        y_true_small = dataset[idx].y.detach().cpu().numpy()
        y_maps_small = np.zeros(partition_res_small.shape, dtype=int)
        for i in range(y_pred_small.shape[0]):
            for t in res_lst_small[i]:
                y_maps_small[t] = y_pred_small[i]

        y_maps = cv2.resize(y_maps_small, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)

        miou = meanIOU(msk_discrete, y_maps)

        acc = np.sum(y_pred_small == y_true_small)/y_pred_small.shape[0]
        _more_log = {}
        if plot:
            print('acc:',np.around(acc,3), 'meanIOU:', np.around(miou, 3))
            plt.imshow(oem.utils.make_rgb(msk_discrete))
            plt.show()
            plt.imshow(oem.utils.make_rgb(y_maps))
            plt.show()
        if log_image:
            res = oem.utils.make_rgb(y_maps)
            _more_log['ymap_rgb'] = res
    return acc, miou, _more_log


def make_prediction(data, chosen_idx, model_gnn, model_feats, device, preprocess,
                    thresh=0.03, cnt_area_max=15, img_small_size=(256, 256), test=False, plot=False):
    
    
        
    assert chosen_idx < len(data)
    model_gnn.eval()
    model_feats.eval()
    with torch.no_grad():    
        img, msk, fn = data[chosen_idx]
        prd_size = img.shape[1:]
        img = np.moveaxis(img.numpy(), 0, -1)
        msk_discrete = np.argmax(msk.numpy(), axis=0)
        msk_discrete = cv2.resize(msk_discrete, dsize=prd_size, interpolation=cv2.INTER_NEAREST)
        msk_discrete_rgb = oem.utils.make_rgb(msk_discrete)
        
        img_small = cv2.resize(img, dsize=img_small_size, interpolation=cv2.INTER_CUBIC)
        msk_discrete_small = cv2.resize(msk_discrete, dsize=img_small_size, interpolation=cv2.INTER_NEAREST)
        
        
        
        dst_small = cv2.Canny(np.uint8(img_small*255), 50, 200, None, 3)
        dst_small = dst_small>0
        
        partition_res_small, mean_c_res_small, res_lst_small, cnt_res_lst_small, num_nodes_small = partition_image(dst_small, img_small, thresh=thresh, cnt_area_max=cnt_area_max)

        partition_res = cv2.resize(partition_res_small, dsize=prd_size, interpolation=cv2.INTER_NEAREST)

        r, s_lst, p_lst = avg_acc(num_nodes_small, partition_res, res_lst_small, msk_discrete)

        V,E = generate_graph_from_partition(partition_res_small)
        
        crop_img_dataset_small = CroppedImageDataset(fn, img_small, partition_res_small, res_lst_small, num_nodes_small, preprocess)
        dataloader = DataLoader(crop_img_dataset_small, batch_size=512, shuffle=False)
        torch.cuda.empty_cache()
        ts_lst = []
        for ts in dataloader:
            with torch.no_grad():
                val = model_feats(ts.to(device))
                ts_lst.append(val.detach().cpu())

        feats = torch.mean(torch.cat(ts_lst, axis=0),axis=(-2,-1)).detach().cpu()

    #     x = mean_c_res # use mean color pixels as the feature for now.
        x = feats
        y = generate_labels_from_graph(res_lst_small, msk_discrete_small, num_nodes_small)
        
        
        edge_index = [[int(u), int(v)] for u in E.keys() for v in E[u]]
        edge_index = torch.tensor(edge_index).T 
        
        data = Data(x=x, edge_index=edge_index, y=y)
        out = model_gnn(data.to(device))
        out = out.detach().cpu().numpy()
        
        y_pred_small = np.argmax(out, axis=1)
        y_true_small = y
        y_maps_small = np.zeros(partition_res_small.shape, dtype=int)
        for i in range(y_pred_small.shape[0]):
            for t in res_lst_small[i]:
                y_maps_small[t] = y_pred_small[i]
        
        y_maps = cv2.resize(y_maps_small, dsize=prd_size, interpolation=cv2.INTER_NEAREST)

        miou = meanIOU(msk_discrete, y_maps)

        acc = np.sum(y_pred_small == y_true_small)/y_pred_small.shape[0]
        
        if plot:
            print('acc:',np.around(acc,3), 'meanIOU:', np.around(miou, 3))
            plt.imshow(oem.utils.make_rgb(msk_discrete))
            plt.show()
            plt.imshow(oem.utils.make_rgb(y_maps))
            plt.show()
            
        
        prd = oem.utils.make_rgb(y_maps)
        
        _log = {}
        _log['prd_size'] = prd_size
        _log['test'] = test
        _log['img'] = img
        _log['msk'] = msk
        _log['fn'] = fn
        _log['msk_discrete'] = msk_discrete
        _log['msk_discrete_rgb'] = msk_discrete_rgb
        _log['img_small'] = img_small
        _log['msk_discrete_small'] = msk_discrete_small
        _log['dst_small'] = dst_small
        
        
        _log['partition_res'] = partition_res
        _log['partition_res_small'] = partition_res_small
        _log['mean_c_res_small'] = mean_c_res_small
        _log['res_lst_small'] = res_lst_small
        _log['cnt_res_lst_small'] = cnt_res_lst_small
        _log['num_nodes_small'] = num_nodes_small
        
        _log['crop_img_dataset_small'] = crop_img_dataset_small
        _log['x'] = x
        _log['y'] = y
        _log['V'] = V
        _log['E'] = E
        _log['edge_index'] = edge_index
        
        _log['out'] = out
        _log['y_pred_small'] = y_pred_small
        _log['y_maps_small'] = y_maps_small
        _log['y_maps'] = y_maps
        _log['prd'] = prd
        
        _log['acc'] = acc
        _log['miou'] = miou
        
        return _log
    