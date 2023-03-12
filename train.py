from pathlib import Path
import pytorch_lightning as pl
from torch.optim import Adam
import torch
import numpy as np
from itertools import combinations
import torch.nn.functional as F
import hydra
from dataset import ShapeNetDataset
from utils.util import real2prob, vote_center, back_filtering, vote_rotation, fibonacci_sphere, remove_ambiguity
import torch.nn as nn
import time
from multiprocessing import cpu_count
from pytorch_lightning import loggers as pl_loggers
from torch import optim
from src_shot.build import shot
from pytorch_lightning.callbacks import ModelCheckpoint

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
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return self.dropout(x + x_res)
    

class BeyondCPPF(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        # 2 for translation, 3 axis, 3 confidence per axis, 3 nocs coord, 3 scale
        fcs = [len(list(combinations(np.arange(cfg.ntuple), 2))) * 4 + cfg.ntuple * 64] + [128] * 5 + [2 + 3 * cfg.rot_num_bins + 3 + 3 + 3]
        self.tuple_encoder = nn.Sequential(
            *[ResLayer(fcs[i], fcs[i + 1], False) for i in range(len(fcs) - 1)]
        )

        fcs_shot = [352,] + [128] * 5 + [64,]
        self.shot_encoder = nn.Sequential(
            *[ResLayer(fcs_shot[i], fcs_shot[i + 1], False) for i in range(len(fcs_shot) - 1)]
        )
        num_samples = int(4 * np.pi / (cfg.eval.angle_tol / 180 * np.pi))
        self.sphere_pts = torch.from_numpy(np.array(fibonacci_sphere(num_samples), dtype=np.float32)).cuda()

    def prepare_tuple_inputs(self, batch):
        pc = batch['pc'][0]
        point_idxs = batch['point_idxs'][0]
        normal = batch['normal'][0]
        point_idx_more = torch.randint(0, pc.shape[0], (point_idxs.shape[0], self.cfg.ntuple - 2), device='cuda')
        point_idx_all = torch.cat([point_idxs, point_idx_more], -1)
        
        shot_feat = self.shot_encoder(batch['shot'][0])
        shot_inputs = torch.cat([shot_feat[point_idx_all[:, i]] for i in range(self.cfg.ntuple)], -1)
        normal_inputs = torch.cat([torch.max(torch.sum(normal[point_idx_all[:, i]] * normal[point_idx_all[:, j]], dim=-1, keepdim=True),
                            torch.sum(-normal[point_idx_all[:, i]] * normal[point_idx_all[:, j]], dim=-1, keepdim=True))
                    for (i, j) in combinations(np.arange(point_idx_all.shape[-1]), 2)], -1)
        coord_inputs = torch.cat([pc[point_idx_all[:, i]] - pc[point_idx_all[:, j]] for (i, j) in combinations(np.arange(point_idx_all.shape[-1]), 2)], -1)
        inputs = torch.cat([coord_inputs, normal_inputs, shot_inputs], -1)
        return inputs, point_idx_all.long()

    def training_step(self, batch, batch_idx):
        inputs, point_idx_all = self.prepare_tuple_inputs(batch)

        preds = self.tuple_encoder(inputs)
        preds_tr = preds[..., :2]
        preds_axis = preds[..., 2:2+3*self.cfg.rot_num_bins].reshape(-1, 3, self.cfg.rot_num_bins)
        preds_conf_axis = preds[..., -9:-6]
        preds_nocs = preds[..., -6:-3]
        preds_scale = preds[..., -3:]

        num_top = int(self.cfg.topk * preds.shape[0])
        loss_tr = torch.mean((preds_tr - batch['targets_tr'][0]) ** 2, -1)
        loss_tr = torch.mean(torch.topk(loss_tr, num_top, largest=False, sorted=False)[0]) * 100.  # not very sensitive to the weight

        loss_scale = torch.mean((preds_scale - batch['targets_scale']) ** 2, -1)
        loss_scale = torch.mean(loss_scale) * 10 # optimize scale loss for all pairs

        loss_nocs = (preds_nocs - batch['nocs'][0][point_idx_all[:, 0]]) ** 2
        loss_nocs = torch.mean(loss_nocs)
        
        loss_axis_all = 0
        for i in range(3):
            pred_axis = preds_axis[:, i]
            target_axis = batch['targets_rot'][0][:, i]
            pred_conf = preds_conf_axis[:, i]
            loss_axis = F.kl_div(F.log_softmax(pred_axis, -1), real2prob(target_axis, np.pi, self.cfg.rot_num_bins, circular=False), reduction='none').sum(-1)
                            
            loss_axis, best_idx = torch.topk(loss_axis, num_top, largest=False, sorted=False)
            loss_axis = torch.mean(loss_axis)
            
            target_conf = torch.zeros((preds.shape[0],)).float().cuda()
            target_conf[best_idx] = 1.
            loss_conf = F.binary_cross_entropy_with_logits(pred_conf, target_conf)
            
            loss_axis_all += loss_axis + loss_conf

        loss = loss_tr + loss_scale + loss_nocs + loss_axis_all
        
        self.log('tr', loss_tr, prog_bar=True)
        self.log('scale', loss_scale, prog_bar=True)
        self.log('nocs', loss_nocs, prog_bar=True)
        self.log('axis', loss_axis_all, prog_bar=True)
        return loss
    
    def forward(self, pc):
        shot_feat = shot.compute(pc, self.cfg.res * 10, self.cfg.res * 10).reshape(-1, 352).astype(np.float32)
        shot_feat[np.isnan(shot_feat)] = 0
            
        normal = shot.estimate_normal(pc, self.cfg.res * 10).reshape(-1, 3).astype(np.float32)
        normal[np.isnan(normal)] = 0

        point_idxs = np.random.randint(0, pc.shape[0], (self.cfg.eval.num_pairs, 2))

        inputs, point_idx_all = self.prepare_tuple_inputs(
            {
                'pc': torch.from_numpy(pc).cuda()[None],
                'point_idxs': torch.from_numpy(point_idxs).cuda()[None],
                'normal': torch.from_numpy(normal).cuda()[None],
                'shot': torch.from_numpy(shot_feat).cuda()[None]
            }
        )

        preds = self.tuple_encoder(inputs)
        preds_tr = preds[..., :2]
        preds_axis = preds[..., 2:2+3*self.cfg.rot_num_bins].reshape(-1, 3, self.cfg.rot_num_bins)
        preds_conf_axis = preds[..., -9:-6]
        preds_nocs = preds[..., -6:-3]
        preds_scale = preds[..., -3:]

        preds_axis = torch.softmax(preds_axis, -1)
        preds_axis = torch.multinomial(preds_axis.reshape(-1, self.cfg.rot_num_bins), 1).float().reshape(-1, 3)
        preds_axis = preds_axis / (self.cfg.rot_num_bins - 1) * np.pi
        preds_up, preds_right, preds_front = preds_axis[:, 0], preds_axis[:, 1], preds_axis[:, 2] 
        
        num_disgard = int((1 - self.cfg.topk) * preds_up.shape[0])
        preds_conf_up, preds_conf_right, preds_conf_front = preds_conf_axis[:, 0], preds_conf_axis[:, 1], preds_conf_axis[:, 2]
        preds_conf_up[torch.topk(preds_conf_up, num_disgard, largest=False, sorted=False)[1]] = 0
        preds_conf_up[preds_conf_up > 0] = 1
        
        preds_conf_right[torch.topk(preds_conf_right, num_disgard, largest=False, sorted=False)[1]] = 0
        preds_conf_right[preds_conf_right > 0] = 1
        
        preds_conf_front[torch.topk(preds_conf_front, num_disgard, largest=False, sorted=False)[1]] = 0
        preds_conf_front[preds_conf_front > 0] = 1
                
        grid_obj, pred_center = vote_center(pc, preds_tr, point_idxs, self.cfg)
        mask = back_filtering(pc, preds_tr, point_idxs, pred_center, grid_obj, self.cfg)
        
        point_idxs = point_idxs[mask]
        preds_tr = preds_tr[mask]
        preds_up = preds_up[mask]
        preds_right = preds_right[mask]
        preds_front = preds_front[mask]
        preds_conf_up = preds_conf_up[mask]
        preds_conf_right = preds_conf_right[mask]
        preds_conf_front = preds_conf_front[mask]
        point_idx_all = point_idx_all[mask]
        preds_nocs = preds_nocs[mask]
        preds_scale = preds_scale[mask]
        
        final_directions = vote_rotation(preds_up, preds_right, preds_conf_up, preds_conf_right, 
                                         pc, point_idxs, self.cfg, self.sphere_pts)
        up = final_directions[0]
        if not self.cfg.up_sym:
            right = final_directions[1]
            right -= np.dot(up, right) * up
            right /= (np.linalg.norm(right) + 1e-9)
        else:
            right = np.array([0, -up[2], up[1]])
            right /= (np.linalg.norm(right) + 1e-9)
        
        pred_scale = preds_scale.mean(0).cpu().numpy()
        pred_coord = preds_nocs.cpu().numpy() * pred_scale.max()
        pc_vote = pc[point_idx_all[:, 0].cpu().numpy()]

        pred_rot = remove_ambiguity(up, right, pc_vote, pred_center, pred_coord, self.cfg)
        return pred_rot, pred_center, pred_scale

    def configure_optimizers(self):
        opt = Adam([*self.shot_encoder.parameters(), *self.tuple_encoder.parameters()], 
                    lr=self.cfg.opt.lr, weight_decay=self.cfg.opt.weight_decay)
        return [opt], [optim.lr_scheduler.StepLR(opt, 25, 0.5)]

@hydra.main(config_path='./config', config_name='config', version_base='1.2')
def main(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    model_ckpt = ModelCheckpoint(save_last=True, save_top_k=0, every_n_epochs=1)
    pl_module = BeyondCPPF(cfg)
    trainer = pl.Trainer(max_epochs=100, gpus=[0], callbacks=[model_ckpt], logger=pl_loggers.TensorBoardLogger(save_dir=output_dir)) # check_val_every_n_epoch=10)
    
    def init_fn(i):
        return np.random.seed(round(time.time() * 1000) % (2 ** 32) +  i)
    
    shapenames = open(hydra.utils.to_absolute_path('data/shapenet_train.txt')).read().splitlines() + open(hydra.utils.to_absolute_path('data/shapenet_val.txt')).read().splitlines()
    shapenames = [line.split()[1] for line in shapenames if int(line.split()[0]) == cfg.category]
    ds = ShapeNetDataset(cfg, shapenames)
    df = torch.utils.data.DataLoader(ds, pin_memory=True, batch_size=1, shuffle=True, num_workers=cpu_count() // 2, worker_init_fn=init_fn)
    trainer.fit(pl_module, df)
    
if __name__ == '__main__':
    main()