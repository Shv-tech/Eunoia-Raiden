"""
Eunoia Raiden v2 (恵雷) — 300M GNN Perception Layer
SHV Groups AGI Research Division
core/gnn_perception.py

512-dim hidden, 4 message-passing layers.
Outputs [N, 512] node embeddings to feed SlotBottleneck.
No bugs found in audit — included verbatim for completeness.
"""

from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

EDGE_ADJACENT   = 0
EDGE_INSIDE     = 1
EDGE_ALIGNED_H  = 2
EDGE_ALIGNED_V  = 3
EDGE_SAME_COLOR = 4
EDGE_SAME_SIZE  = 5
NUM_EDGE_TYPES  = 6


class GridParser:
    @staticmethod
    def _flood_fill(grid: np.ndarray) -> List[dict]:
        rows, cols = grid.shape
        visited    = np.zeros((rows, cols), dtype=bool)
        objects    = []
        oid        = 1
        for r in range(rows):
            for c in range(cols):
                color = int(grid[r, c])
                if color == 0 or visited[r, c]:
                    continue
                cells = []
                q     = deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if (0<=nr<rows and 0<=nc<cols
                                and not visited[nr,nc]
                                and grid[nr,nc]==color):
                            visited[nr,nc]=True
                            q.append((nr,nc))
                cells_arr = np.array(cells, dtype=np.int32)
                objects.append({
                    "id": oid, "color": color, "cells": cells_arr,
                    "bbox": (int(cells_arr[:,0].min()), int(cells_arr[:,1].min()),
                             int(cells_arr[:,0].max()), int(cells_arr[:,1].max())),
                })
                oid += 1
        return objects

    @staticmethod
    def _node_features(obj: dict, H: int, W: int) -> np.ndarray:
        cells = obj["cells"]
        color = obj["color"]
        min_r, min_c, max_r, max_c = obj["bbox"]
        n     = len(cells)
        bh    = max_r-min_r+1; bw = max_c-min_c+1
        cr    = float(cells[:,0].mean()); cc = float(cells[:,1].mean())
        cset  = set(map(tuple, cells.tolist()))
        peri  = sum(1 for r,c in cells
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                    if (r+dr,c+dc) not in cset)
        aspect = min((bh/max(bw,1)),5.0)/5.0

        mask = np.zeros((bh,bw),dtype=np.float32)
        for r,c in cells: mask[r-min_r,c-min_c]=1.0
        T=6; hg=np.zeros((T,T),dtype=np.float32)
        for i in range(T):
            for j in range(T):
                r0=int(i*bh/T); r1=max(int((i+1)*bh/T),r0+1); r1=min(r1,bh)
                c0=int(j*bw/T); c1=max(int((j+1)*bw/T),c0+1); c1=min(c1,bw)
                b=mask[r0:r1,c0:c1]; hg[i,j]=b.max() if b.size>0 else 0.0

        feat = np.zeros(128, dtype=np.float32)
        if 0<=color<=9: feat[color]=1.0
        feat[10]=n/max(H*W,1); feat[11]=cr/max(H-1,1); feat[12]=cc/max(W-1,1)
        feat[13]=min_r/max(H-1,1); feat[14]=max_r/max(H-1,1)
        feat[15]=min_c/max(W-1,1); feat[16]=max_c/max(W-1,1)
        feat[17]=aspect; feat[18]=peri/max(4*n,1); feat[19]=n/max(bh*bw,1)
        feat[20:56]=hg.flatten()
        return feat

    @staticmethod
    def _build_edges(objects: List[dict], H: int, W: int):
        n=len(objects); el=[]; tl=[]
        for i in range(n):
            oi=objects[i]; ci=set(map(tuple,oi["cells"].tolist()))
            ni=len(ci); mri,mci,xri,xci=oi["bbox"]
            cri=oi["cells"][:,0].mean(); cci=oi["cells"][:,1].mean()
            for j in range(i+1,n):
                oj=objects[j]; cj=set(map(tuple,oj["cells"].tolist()))
                nj=len(cj); mrj,mcj,xrj,xcj=oj["bbox"]
                crj=oj["cells"][:,0].mean(); ccj=oj["cells"][:,1].mean()
                adj=any((r+dr,c+dc) in cj
                        for r,c in ci
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])
                i_in_j=(mri>=mrj and xri<=xrj and mci>=mcj and xci<=xcj)
                j_in_i=(mrj>=mri and xrj<=xri and mcj>=mci and xcj<=xci)
                for cond,et in [
                    (adj, EDGE_ADJACENT),
                    (i_in_j or j_in_i, EDGE_INSIDE),
                    (abs(cri-crj)<1.0, EDGE_ALIGNED_H),
                    (abs(cci-ccj)<1.0, EDGE_ALIGNED_V),
                    (oi["color"]==oj["color"], EDGE_SAME_COLOR),
                    (min(ni,nj)/max(max(ni,nj),1)>=0.9, EDGE_SAME_SIZE),
                ]:
                    if cond:
                        el+=[(i,j),(j,i)]; tl+=[et,et]
        return el,tl

    @classmethod
    def parse(cls, grid: np.ndarray,
              device: torch.device = torch.device("cpu")
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid    = np.asarray(grid, dtype=np.uint8)
        H, W    = grid.shape
        objects = cls._flood_fill(grid)
        if not objects:
            return (torch.zeros(1,128,dtype=torch.float32,device=device),
                    torch.zeros(2,0,dtype=torch.long,device=device),
                    torch.zeros(0,dtype=torch.long,device=device))
        nf = torch.tensor(
            np.stack([cls._node_features(o,H,W) for o in objects]),
            dtype=torch.float32, device=device)
        el, tl = cls._build_edges(objects,H,W)
        if el:
            ei=torch.tensor(el,dtype=torch.long,device=device).t().contiguous()
            et=torch.tensor(tl,dtype=torch.long,device=device)
        else:
            ei=torch.zeros(2,0,dtype=torch.long,device=device)
            et=torch.zeros(0,dtype=torch.long,device=device)
        return nf, ei, et


class HeterogeneousMessagePassing(nn.Module):
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(NUM_EDGE_TYPES)
        ])
        self.update_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm       = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_types):
        N, D   = x.shape
        agg    = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            for t in range(NUM_EDGE_TYPES):
                mask = (edge_types==t)
                if not mask.any(): continue
                mi  = torch.cat([x[src[mask]], x[dst[mask]]], dim=-1)
                msg = self.message_mlps[t](mi)
                agg.scatter_add_(0, dst[mask].unsqueeze(-1).expand_as(msg), msg)
        return self.norm(x + self.update_gru(agg, x))


class GNNPerception(nn.Module):
    """
    Input  : (node_features [N,128], edge_index [2,E], edge_types [E])
    Output : [N, 512]
    """
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 512,
                 num_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            HeterogeneousMessagePassing(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, node_features, edge_index, edge_types):
        x = self.input_proj(node_features)
        for layer in self.layers:
            x = layer(x, edge_index, edge_types)
        return x