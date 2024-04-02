from pathlib import Path
import pytorch_lightning as pl
from torch.optim import Adam
import torch
import numpy as np
from itertools import combinations
import torch.nn.functional as F
import hydra
from dataset import ShapeNetDirectDataset
import torch.nn as nn
import time
from multiprocessing import cpu_count
from pytorch_lightning import loggers as pl_loggers
from torch import optim
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.util import real2prob, prob2real


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


class BeyondCPPF(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        # self.vis = Visdom(port=12345)
        
        fcs_shot = [352,] + [128] * 5 + [64,]
        self.shot_encoder = nn.Sequential(
            *[ResLayer(fcs_shot[i], fcs_shot[i + 1], False) for i in range(len(fcs_shot) - 1)]
        )
        
        input_dim = len(list(combinations(np.arange(cfg.num_more + 2), 2))) * 4 + (cfg.num_more + 2) * 64
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

    def prepare_tuple_inputs(self, points, point_idxs_all, shot_feat, normal):
        shot_inputs = torch.cat([shot_feat[point_idxs_all[:, i]] for i in range(0, self.cfg.num_more + 2)], -1)
        normal_inputs = torch.cat([torch.max(torch.sum(normal[point_idxs_all[:, i]] * normal[point_idxs_all[:, j]], dim=-1, keepdim=True),
                            torch.sum(-normal[point_idxs_all[:, i]] * normal[point_idxs_all[:, j]], dim=-1, keepdim=True))
                    for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], -1)

        coord_inputs = torch.cat([points[point_idxs_all[:, i]] - points[point_idxs_all[:, j]] for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], -1)
        inputs = torch.cat([coord_inputs, normal_inputs, shot_inputs], -1)
        return inputs

    def training_step(self, batch, batch_idx):
        pc_canon = batch['pc_canon'][0]
        point_idxs_all = batch['point_idxs_all'][0]
        points = batch['pc'][0]
        shot_feat = self.shot_encoder(batch['shot'][0])
        # shot_feat.fill_(0)
        normal = batch['normal'][0]
        inputs = self.prepare_tuple_inputs(points,  point_idxs_all, shot_feat, normal)
        # pca_quat = matrix_to_quaternion(pca_basis)

        feat = self.tuple_encoder(inputs)
        preds_cls = self.logit_encoder(feat).reshape(feat.shape[0], 6, -1)
        with torch.no_grad():
            target_cls = real2prob(torch.clamp(pc_canon[point_idxs_all[:, :2]], -0.5, 0.5) + 0.5, 1., preds_cls.shape[-1]).reshape(feat.shape[0], 6, -1)  # 2NP x 3 x NBIN
        loss_cls = F.kl_div(F.log_softmax(preds_cls, dim=-1), target_cls, reduction='batchmean')
        
        preds_scale = self.scale_encoder(feat)
        target_scale = batch['bound'][0]
        loss_scale = F.mse_loss(preds_scale, target_scale[None].expand_as(preds_scale))
        loss = loss_cls + loss_scale
        # self.log('loss', loss, prog_bar=True)
        self.log('cls', loss_cls, prog_bar=True)
        self.log('scale', loss_scale, prog_bar=True)
        # vis_weights = torch_scatter.scatter_add(preds_weight[:, 1:2].expand_as(point_idxs_all).reshape(-1), point_idxs_all.reshape(-1), dim_size=points.shape[0]).detach()
        # vis_weights /= vis_weights.max()
        
        # if batch_idx % 10 == 0:
        #     # print(vis_weights.max(), vis_weights.mean(), (vis_weights > 0).sum())
        #     cmap = cm.get_cmap('jet')
        #     self.vis.scatter(points.cpu().numpy(), win=4, opts=dict(markersize=3, markercolor=(np.array([cmap(w)[:3] for w in vis_weights.cpu().numpy()]) * 255.).astype(np.uint8)))
        return loss
    
    def forward(self, points, point_idxs_all, shot_feat, normal):
        inputs = self.prepare_tuple_inputs(points, point_idxs_all, self.shot_encoder(shot_feat), normal)
        feat = self.tuple_encoder(inputs)
        preds_scale = self.scale_encoder(feat)
        preds_cls = self.logit_encoder(feat).reshape(feat.shape[0], 6, -1)
        return preds_cls, preds_scale

    def configure_optimizers(self):
        opt = Adam([*self.shot_encoder.parameters(),
                    *self.tuple_encoder.parameters(),
                    *self.logit_encoder.parameters(),
                    *self.scale_encoder.parameters()], 
                    lr=self.cfg.opt.lr, weight_decay=self.cfg.opt.weight_decay)
        return [opt], [optim.lr_scheduler.StepLR(opt, 25, 0.5)]

import wandb
@hydra.main(config_path='./config', config_name='config', version_base='1.2')
def train(cfg):
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    model_ckpt = ModelCheckpoint(save_last=True, save_top_k=-1, every_n_epochs=10, filename='{epoch}')
    pl_module = BeyondCPPF(cfg)
    trainer = pl.Trainer(max_epochs=101, accelerator='auto', callbacks=[model_ckpt], 
                         logger=pl_loggers.TensorBoardLogger(save_dir=output_dir),
                         detect_anomaly=False) # check_val_every_n_epoch=10)
    
    def init_fn(i):
        return np.random.seed(round(time.time() * 1000) % (2 ** 32) +  i)

    ds = ShapeNetDirectDataset(cfg)
    df = torch.utils.data.DataLoader(ds, pin_memory=True, batch_size=1, shuffle=True, num_workers=cpu_count() // 2, worker_init_fn=init_fn)
    trainer.fit(pl_module, df)
    
    
if __name__ == '__main__':
    train()