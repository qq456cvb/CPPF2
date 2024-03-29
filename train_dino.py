from pathlib import Path
import pytorch_lightning as pl
from torch.optim import Adam
import torch
import numpy as np
from itertools import combinations, permutations
import torch.nn.functional as F
import hydra
from dataset import ShapeNetExportDataset, id2category
import torch.nn as nn
import time
from multiprocessing import cpu_count
from pytorch_lightning import loggers as pl_loggers
from torch import optim
from src_shot.build import shot
from pytorch_lightning.callbacks import ModelCheckpoint
import torch_scatter
from utils.util import real2prob, prob2real
from torch.nn import MultiheadAttention

class ResLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bn=False, dropout=False) -> None:
        super().__init__()
        # assert(bn is False)
        self.fc1 = torch.nn.Linear(dim_in, dim_out)
        if bn:
            self.bn1 = torch.nn.BatchNorm1d(dim_out)
        else:
            self.bn1 = lambda x: x
        self.fc2 = torch.nn.Linear(dim_out, dim_out)
        if bn:
            self.bn2 = torch.nn.BatchNorm1d(dim_out)
        else:
            self.bn2 = lambda x: x
        if dim_in != dim_out:
            self.fc0 = torch.nn.Linear(dim_in, dim_out)
        else:
            self.fc0 = None
        self.dropout = nn.Dropout(0.2) if dropout else nn.Identity()
    
    def forward(self, x):
        x_res = x if self.fc0 is None else self.fc0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return self.dropout(x + x_res)


class Attention(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.2):
        super().__init__()
        self.attn = MultiheadAttention(dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, *x):
        return self.layer_norm(self.attn(*x)[0] + x[0])
    

class BeyondCPPF(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        # self.vis = Visdom(port=12345)
        
        transformer_dim = 256
        input_dim = len(list(combinations(np.arange(cfg.num_more + 2), 2))) * 3 + transformer_dim
        output_dim = 256  # 3 for scale
        fcs = [input_dim] + [128] * 5 + [output_dim]
        
        self.tuple_encoder = nn.Sequential(
                *[ResLayer(fcs[i], fcs[i + 1], False) for i in range(len(fcs) - 1)]
        )
        self.logit_encoder = nn.Sequential(
            ResLayer(256, 256, False),
            ResLayer(256, 256, False),
            ResLayer(256, 64 * 3, False),
        )
        self.scale_encoder = nn.Sequential(
            ResLayer(256, 128, False),
            ResLayer(128, 64, False),
            ResLayer(64, 3, False),
        )
        # self.to_q = nn.Linear(3, transformer_dim)
        # self.to_kv = nn.Linear(3, transformer_dim)
        self.desc_transform = nn.Linear(1024, transformer_dim)
        self.desc_pair_transform = nn.Linear(transformer_dim * (self.cfg.num_more + 2), transformer_dim)
        # self.attns = nn.ModuleList([
        #     Attention(transformer_dim) for _ in range(12)
        # ])
        # self.out_transform = nn.Linear(transformer_dim * len(list(combinations(np.arange(cfg.num_more + 2), 2))), transformer_dim)

    def prepare_tuple_inputs(self, points, point_descs, point_idxs_all):
        coord_inputs = torch.cat([points[point_idxs_all[:, i]] - points[point_idxs_all[:, j]] for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], -1)  # N x C(k, 2) x 3
        # desc_inputs = torch.stack([point_descs[point_idxs_all[:, i]] - point_descs[point_idxs_all[:, j]] for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], 1)  # N x C(k, 2) x 3
        # inputs = torch.cat([coord_inputs, desc_inputs], -1)
        desc_inputs = torch.cat([self.desc_transform(point_descs[point_idxs_all[:, i]]) for i in range(0, self.cfg.num_more + 2)], -1)
        inputs = torch.cat([coord_inputs, self.desc_pair_transform(desc_inputs)], -1)
        return inputs

    def training_step(self, batch, batch_idx):
        pc_canon = batch['pc_canon'][0]
        points = batch['pc'][0]
        point_descs = batch['desc'][0]
        point_idxs_all = torch.from_numpy(np.random.randint(0, points.shape[0], (10000, self.cfg.num_more + 2))).long().cuda()
        inputs = self.prepare_tuple_inputs(points, point_descs,  point_idxs_all)

        feat = self.tuple_encoder(inputs)
        preds_cls = self.logit_encoder(feat).reshape(feat.shape[0], 6, -1)
        with torch.no_grad():
            target_cls = real2prob(torch.clamp(pc_canon[point_idxs_all[:, :2]], -0.5, 0.5) + 0.5, 1., preds_cls.shape[-1]).reshape(feat.shape[0], 6, -1)  # 2NP x 3 x NBIN
        loss_cls = F.kl_div(F.log_softmax(preds_cls, dim=-1), target_cls, reduction='batchmean')
        
        preds_scale = self.scale_encoder(feat)
        target_scale = batch['bound'][0]
        loss_scale = F.mse_loss(preds_scale, target_scale[None].expand_as(preds_scale))
        loss = loss_cls + loss_scale
        self.log('loss', loss, prog_bar=True)
        self.log('cls', loss_cls, prog_bar=True)
        self.log('scale', loss_scale, prog_bar=True)
        # vis_weights = torch_scatter.scatter_add(preds_weight[:, 1:2].expand_as(point_idxs_all).reshape(-1), point_idxs_all.reshape(-1), dim_size=points.shape[0]).detach()
        # vis_weights /= vis_weights.max()
        
        # if batch_idx % 10 == 0:
        #     # print(vis_weights.max(), vis_weights.mean(), (vis_weights > 0).sum())
        #     cmap = cm.get_cmap('jet')
        #     self.vis.scatter(points.cpu().numpy(), win=4, opts=dict(markersize=3, markercolor=(np.array([cmap(w)[:3] for w in vis_weights.cpu().numpy()]) * 255.).astype(np.uint8)))
        return loss
    
    def forward(self, points, point_descs, point_idxs_all):
        inputs = self.prepare_tuple_inputs(points, point_descs, point_idxs_all)
        feat = self.tuple_encoder(inputs)
        preds_scale = self.scale_encoder(feat)
        preds_cls = self.logit_encoder(feat).reshape(feat.shape[0], 6, -1)
        return preds_cls, preds_scale

    def configure_optimizers(self):
        opt = Adam(self.parameters(), 
                    lr=self.cfg.opt.lr, weight_decay=self.cfg.opt.weight_decay)
        return [opt], [optim.lr_scheduler.StepLR(opt, 25, 0.5)]

import wandb
import os
@hydra.main(config_path='./config', config_name='config', version_base='1.2')
# aa version, direct use relative coordinates, dropping any local contexts that may be noisy
def train(cfg):
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    model_ckpt = ModelCheckpoint(save_last=True, save_top_k=-1, every_n_epochs=10, filename='{epoch}')
    
    pl_module = BeyondCPPF(cfg=cfg)
    trainer = pl.Trainer(max_epochs=101, accelerator='auto', callbacks=[model_ckpt], 
                         logger=pl_loggers.TensorBoardLogger(save_dir=output_dir),
                         detect_anomaly=False) # check_val_every_n_epoch=10)
    
    def init_fn(i):
        return np.random.seed(round(time.time() * 1000) % (2 ** 32) +  i)

    ds = ShapeNetExportDataset(cfg)
    df = torch.utils.data.DataLoader(ds, pin_memory=True, batch_size=1, shuffle=True, num_workers=cpu_count() // 2, worker_init_fn=init_fn)
    trainer.fit(pl_module, df)

from omegaconf import OmegaConf
import trimesh
from dataset import generate_target_pairs, generate_target_noaux, shapenet_obj_scales, rotx, roty, rotz
import cv2
from tqdm import tqdm
from utils.util import fibonacci_sphere
from scipy.spatial.transform import Rotation as R

def vote_center(pc, preds_tr, res, point_idxs, num_rots=36, vis=None):
    corners = torch.stack([pc.min(0)[0], pc.max(0)[0]])
    grid_res = ((corners[1] - corners[0]) / res).long() + 1
    grid_obj = torch.zeros([*grid_res]).to(pc)
    
    proj_len = preds_tr[:, 0]
    odist = preds_tr[:, 1]
    pairs = pc[point_idxs[:, :2]]
    a = pairs[:, 0]
    b = pairs[:, 1]
    ab = a - b
    mask = (torch.norm(ab, dim=-1) > 1e-7) & (odist > res)
    pairs = None
    proj_len, odist, a, b, ab = proj_len[mask], odist[mask], a[mask], b[mask], ab[mask]
    ab = ab / torch.clamp_min(torch.norm(ab, dim=-1, keepdim=True), 1e-7)
    c = a - ab * proj_len[..., None]
    co = torch.stack([torch.zeros((ab.shape[0],)).to(pc), -ab[..., 2], ab[..., 1]], -1)
    invalid = torch.norm(co, dim=-1) < 1e-7
    co[invalid] = torch.stack([-ab[invalid][..., 1], ab[invalid][..., 0], torch.zeros((ab[invalid].shape[0],)).to(pc)], -1)
    
    x = co / torch.norm(co, dim=-1, keepdim=True) * odist[..., None]
    y = torch.cross(x, ab, dim=-1)
    
    # TODO: adaptive num rots
    angles = torch.arange(num_rots).to(pc) / num_rots * 2 * np.pi
    offset = torch.cos(angles[None, :, None]) * x[:, None] + torch.sin(angles[None, :, None]) * y[:, None]  # n x numrot x 3
    center_grid = (c[:, None] + offset - corners[0]) / res
    center_grid = (center_grid + 0.5).long().reshape(-1, 3)
    
    valid = torch.all(center_grid > 0, -1) & torch.all(center_grid < grid_res, -1)
    center_grid = center_grid[valid]
    
    center_grid_1d = center_grid[:, 0] * grid_res[1] * grid_res[2] + center_grid[:, 1] * grid_res[2] + center_grid[:, 2]
    grid_obj = torch_scatter.scatter_add(torch.ones_like(center_grid_1d), center_grid_1d, dim_size=grid_res[0] * grid_res[1] * grid_res[2]).reshape(*grid_res)
    
    grid_obj = grid_obj.cpu().numpy()
    if vis is not None:
        vis.heatmap(cv2.rotate(grid_obj.max(0), cv2.ROTATE_90_COUNTERCLOCKWISE), win='33', opts=dict(title='front'))
        vis.heatmap(cv2.rotate(grid_obj.max(1), cv2.ROTATE_90_COUNTERCLOCKWISE), win='34', opts=dict(title='bird'))
        vis.heatmap(cv2.rotate(grid_obj.max(2), cv2.ROTATE_90_COUNTERCLOCKWISE), win='35', opts=dict(title='side'))

    cand = np.array(np.unravel_index([np.argmax(grid_obj, axis=None)], grid_obj.shape)).T[::-1][0]
    cand_world = corners[0].cpu().numpy() + cand * res
    # print(cand, cand_world)
    return grid_obj, cand_world


def vote_rotation(pc, preds_rot, point_idxs, num_rots=36):
    pairs = pc[point_idxs[:, :2]]
    a = pairs[:, 0]
    b = pairs[:, 1]
    ab = a - b
    mask = torch.norm(ab, dim=-1) > 1e-7
    pairs = None
    a, b, ab, preds_rot = a[mask], b[mask], ab[mask], preds_rot[mask]
    ab = ab / torch.clamp_min(torch.norm(ab, dim=-1, keepdim=True), 1e-7)
    co = torch.stack([torch.zeros((ab.shape[0],)).to(pc), -ab[..., 2], ab[..., 1]], -1)
    invalid = torch.norm(co, dim=-1) < 1e-7
    co[invalid] = torch.stack([-ab[invalid][..., 1], ab[invalid][..., 0], torch.zeros((ab[invalid].shape[0],)).to(pc)], -1)
    
    x = co / torch.clamp_min(torch.norm(co, dim=-1, keepdim=True), 1e-7)
    y = torch.cross(x, ab, dim=-1)
    angles = torch.arange(num_rots).to(pc) / num_rots * 2 * np.pi
    offset = torch.cos(angles[None, :, None]) * x[:, None] + torch.sin(angles[None, :, None]) * y[:, None]  # n x numrot x 3
    tan = torch.tan(preds_rot)
    up = tan[:, None, None] * offset + torch.where(tan > 0, 1., -1.)[:, None, None] * ab[:, None]
    up = up / torch.clamp_min(torch.norm(up, dim=-1, keepdim=True), 1e-7)
    
    return up, mask

    
if __name__ == '__main__':
    train()