import sys
import os
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch.optim as optim


def board_one_hot(board_np):
    H, W = board_np.shape
    b = torch.tensor((board_np == 1).astype(np.float32)).view(1, 1, H, W)
    w = torch.tensor((board_np == 2).astype(np.float32)).view(1, 1, H, W)
    e = torch.tensor((board_np == 0).astype(np.float32)).view(1, 1, H, W)
    return b, w, e

def _conv_hits(x, k):
    return F.conv2d(x, k, stride=1, padding=0)

class MotifConv2D(nn.Module):
    def __init__(self):
        super().__init__()
        diag = torch.tensor([[1., 0.], [0., 1.]]).view(1,1,2,2)
        self.register_buffer('k_diag', diag)
        anti = torch.tensor([[0., 1.], [1., 0.]]).view(1,1,2,2)
        self.register_buffer('k_anti', anti)
        all_2=torch.ones(1,1,2,2)
        self.register_buffer('k_all2', all_2)
        cross=torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]]).view(1,1,3,3)
        self.register_buffer('k_cross', cross)
        center=torch.zeros_like(cross)
        center[0,0,1,1]=1.
        self.register_buffer('k_center', center)
        d_e=torch.tensor([[0,0],[0,1]]).view(1,1,2,2).float()
        a_e=torch.tensor([[1,0],[0,0]]).view(1,1,2,2).float()
        b_e=torch.tensor([[0,1],[0,0]]).view(1,1,2,2).float()
        c_e=torch.tensor([[0,0],[1,0]]).view(1,1,2,2).float()
        self.register_buffer('tmp_de',d_e)
        self.register_buffer('tmp_ae',a_e)
        self.register_buffer('tmp_be',b_e)
        self.register_buffer('tmp_ce',c_e)

    @torch.no_grad()
    def forward(self, board_np):
        H, W = board_np.shape
        if H < 3 or W < 3: return {}
        B, Wc, E = board_one_hot(board_np)
        def sq(x): 
            return x.to(torch.bool).squeeze(0).squeeze(0)
        hits = {}
        b_d=_conv_hits(B,self.k_diag)
        e_a=_conv_hits(E,self.k_anti)
        b_a=_conv_hits(B,self.k_anti)
        e_d=_conv_hits(E,self.k_diag)
        hits['bamboo_black']=sq(((b_d==2)&(e_a==2))|((b_a==2)&(e_d==2)))
        w_d=_conv_hits(Wc,self.k_diag)
        w_a=_conv_hits(Wc,self.k_anti)
        hits['bamboo_white']=sq(((w_d==2)&(e_a==2))|((w_a==2)&(e_d==2)))
        b_s2=_conv_hits(B,self.k_all2)
        e_s2=_conv_hits(E,self.k_all2)
        eD=_conv_hits(E,self.tmp_de)
        eA=_conv_hits(E,self.tmp_ae)
        eB=_conv_hits(E,self.tmp_be)
        eC=_conv_hits(E,self.tmp_ce)
        hits['tiger_black']=sq((b_s2>=2)&((eD==1)|(eA==1)|(eB==1)|(eC==1)))
        w_s2=_conv_hits(Wc,self.k_all2)
        hits['tiger_white']=sq((w_s2>=2)&((eD==1)|(eA==1)|(eB==1)|(eC==1)))
        hits['empty_triangle_black']=sq((b_s2==3)&(e_s2==1))
        hits['empty_triangle_white']=sq((w_s2==3)&(e_s2==1))
        e_c=_conv_hits(E,self.k_center)
        b_c=_conv_hits(B,self.k_cross)
        w_c=_conv_hits(Wc,self.k_cross)
        hits['eye_black']=sq((e_c==1)&(b_c==4))
        hits['eye_white']=sq((e_c==1)&(w_c==4))
        return hits

def lattice_edge_index(H, W):
    edges = []
    def nid(i, j): return i * W + j
    for i in range(H):
        for j in range(W):
            u = nid(i, j)
            if i+1 < H: 
                edges.extend([(u,nid(i+1,j)),(nid(i+1,j),u)])
            if j+1 < W: 
                edges.extend([(u,nid(i,j+1)),(nid(i,j+1),u)])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def motif_adjacency_from_hits(hits, H, W):
    n = H * W
    A = torch.zeros(n, n, dtype=torch.float32)
    if not hits: 
        return A
    def nid(i, j): 
        return i * W + j
    def add_2x2(i,j,w):
        cells=[(i,j),(i,j+1),(i+1,j),(i+1,j+1)]
        idxs=[nid(x,y) for x,y in cells if 0<=x<H and 0<=y<W]
        for u in idxs:
            for v in idxs:
                if u!=v: A[u,v]+=w
    def add_eye3x3(i,j,w):
        c=(i+1,j+1)
        ring=[(c[0]-1,c[1]),(c[0]+1,c[1]),(c[0],c[1]-1),(c[0],c[1]+1)]
        idxs=[nid(x,y) for x,y in ring if 0<=x<H and 0<=y<W]
        for u in idxs:
            for v in idxs:
                if u!=v: 
                    A[u,v]+=w
    weights = {'bamboo':1.2, 'tiger':1.0, 'empty_triangle':0.8}
    for k,w in weights.items():
        for c in ['black','white']:
            m = hits[f'{k}_{c}']
            I, J = torch.where(m)
            for i, j in zip(I.tolist(), J.tolist()): 
                add_2x2(i, j, w)
    for c in ['black','white']:
        m = hits[f'eye_{c}']
        I, J = torch.where(m)
        for i, j in zip(I.tolist(), J.tolist()): 
            add_eye3x3(i, j, 1.5)
    return torch.maximum(A, A.t())
