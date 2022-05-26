from glob import glob

import omegaconf
from utils.util import backproject, dilate_mask, fibonacci_sphere, real2prob
import open3d as o3d
from os import cpu_count
import hydra
import torch
from models.utils import smooth_l1_loss
from models.model import PPFEncoder, PointEncoderRaw, ResLayer
from articulated.dataset import ArticulatedDataset, generate_target_rot, generate_target_tr
import numpy as np
import logging
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import visdom
import cupy as cp
from models.voting import ppf_kernel, rot_voting_kernel, backvote_kernel
from multiprocessing import cpu_count
import torch.nn.functional as F
import MinkowskiEngine as ME
import cv2
import pickle
import os
import matplotlib.pyplot as plt
from itertools import combinations


vis = visdom.Visdom(port=21391)


def visualize(vis, *pcs, **opts):
    vis_pc = np.concatenate(pcs)
    vis_label = np.ones((sum([p.shape[0] for p in pcs])), np.int64)
    a = 0
    for i, pc in enumerate(pcs):
        vis_label[a:a+pc.shape[0]] = i + 1
        a += pc.shape[0]
    vis.scatter(vis_pc, vis_label, **opts)
    

def validation(vertices, outputs, probs, res, point_idxs, n_ppfs, num_rots=36, visualize=True):
    with cp.cuda.Device(0):
        block_size = (point_idxs.shape[0] + 512 - 1) // 512

        corners = np.stack([np.min(vertices, 0), np.max(vertices, 0)])
        grid_res = ((corners[1] - corners[0]) / res).astype(np.int32) + 1
        grid_obj = cp.asarray(np.zeros(grid_res, dtype=np.float32))
        ppf_kernel(
            (block_size, 1, 1),
            (512, 1, 1),
            (
                cp.asarray(vertices).astype(cp.float32), cp.asarray(outputs).astype(cp.float32), cp.asarray(probs).astype(cp.float32), cp.asarray(point_idxs).astype(cp.int32), grid_obj, cp.asarray(corners[0]), cp.float32(res), 
                n_ppfs, num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2]
            )
        )
        
        grid_obj = grid_obj.get()
        
        # cand = np.array(np.unravel_index([np.argmax(grid_obj, axis=None)], grid_obj.shape)).T[::-1]
        # grid_obj[cand[-1][0]-20:cand[-1][0]+20, cand[-1][1]-20:cand[-1][1]+20, cand[-1][2]-20:cand[-1][2]+20] = 0
        if visualize:
            vis.heatmap(cv2.rotate(grid_obj.max(0), cv2.ROTATE_90_COUNTERCLOCKWISE), win=3, opts=dict(title='front'))
            vis.heatmap(cv2.rotate(grid_obj.max(1), cv2.ROTATE_90_COUNTERCLOCKWISE), win=4, opts=dict(title='bird'))
            vis.heatmap(cv2.rotate(grid_obj.max(2), cv2.ROTATE_90_COUNTERCLOCKWISE), win=5, opts=dict(title='side'))

        cand = np.array(np.unravel_index([np.argmax(grid_obj, axis=None)], grid_obj.shape)).T[::-1]
        cand_world = corners[0] + cand * res
        # print(cand_world[-1])
        return grid_obj, cand_world


def align_to(a, b):
    a, b = torch.broadcast_tensors(a, b)
    c = torch.cross(a, b, -1)
    zeros = torch.zeros_like(c[..., 0])
    C = torch.stack([zeros, -c[..., 2], c[..., 1], c[..., 2], zeros, -c[..., 0], -c[..., 1], c[..., 0], zeros], -1).reshape(*c.shape[:-1], 3, 3)
    mat = C + ((1 - torch.sum(a * b, -1)) / (torch.sum(c * c, -1) + 1e-9))[..., None, None] * C @ C
    mat[..., 0, 0] += 1
    mat[..., 1, 1] += 1
    mat[..., 2, 2] += 1
    return mat


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
@hydra.main(config_path='config', config_name='config')
def main(cfg):
    EVAL_ONLY = False
    # sample_points('02876657/688eacb56a16702f9a6e43b878d5b335')
    # exit()
    logger = logging.getLogger(__name__)

    
    if EVAL_ONLY:
        path = 'outputs/04-19/22-04'  # not good for right, but good for up
        nepoch = '195'
        cfg = omegaconf.OmegaConf.load(hydra.utils.to_absolute_path(f"{path}/.hydra/config.yaml"))
    ds = ArticulatedDataset('/home/neil/ppf_matching/articulated/data/eyeglasses1', cfg)
    df = torch.utils.data.DataLoader(ds, pin_memory=True, batch_size=cfg.batch_size, shuffle=True, num_workers=10)
    assert cfg.batch_size == 1
    
    topk = cfg.topk
    logger.info('Train')
    best_loss = np.inf
    
    angle_tol = 1.5
    num_samples = int(4 * np.pi / (angle_tol / 180 * np.pi))
    sphere_pts = np.array(fibonacci_sphere(num_samples))
    num_pairs = 1000000
    num_rots = 120
    bmm_size = num_rots * 1000
    n_threads = 512
    
    num_more = cfg.num_more
    fcs = [len(list(combinations(np.arange(num_more + 2), 2))) * 3] + [128] * 10 + [2 + 1 + 1]
    encoder = nn.Sequential(
        *[ResLayer(fcs[i], fcs[i + 1], False, dropout=False if i < len(fcs) - 2 else False) for i in range(len(fcs) - 1)]
    ).cuda()
    
    if EVAL_ONLY:
        encoder.load_state_dict(torch.load(hydra.utils.to_absolute_path(f'{path}/encoder_epoch{nepoch}.pth')))

    opt = optim.Adam(encoder.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(opt, 25, 0.5)
    bce = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(cfg.max_epoch):
        n = 0
        
        loss_meter = AverageMeter()
        loss_tr_meter = AverageMeter()
        loss_up_meter = AverageMeter()
        loss_up_aux_meter = AverageMeter()
        loss_right_meter = AverageMeter()
        loss_scale_meter = AverageMeter()
        loss_nocs_meter = AverageMeter()
        encoder.train()
        with tqdm(df) as t:
            if not EVAL_ONLY:
                for pcs, pc_canons, targets_tr, targets_rot, point_idxs, _, _ in t:
                    # break
                    pc, pc_canon, targets_tr, targets_rot, point_idxs = \
                        pcs.cuda()[0], pc_canons.cuda()[0], targets_tr.cuda()[0], targets_rot.cuda()[0], point_idxs.cuda()[0]
                    
                    opt.zero_grad()
                    
                    point_idx_more = torch.randint(0, pc.shape[0], (point_idxs.shape[0], num_more), device='cuda')
                    point_idx_all = torch.cat([point_idxs, point_idx_more], -1)
                    inputs = torch.stack([pc[point_idx_all[:, i]] - pc[point_idx_all[:, j]] for (i, j) in combinations(np.arange(point_idx_all.shape[-1]), 2)], -2)  # B x N x 3

                    ab = inputs[..., :3]
                    mat = align_to(ab / (ab.norm(dim=-1, keepdim=True) + 1e-9), torch.tensor([1, 0, 0]).to(ab))
                    # x = (mat @ ab[..., None])[..., 0]
                    inputs = (mat @ inputs[..., None])
                    inputs = inputs.reshape(inputs.shape[0], -1)
                    
                    preds = encoder(inputs)
                    preds_tr = preds[..., :2]
                    preds_up = preds[..., 2]
                    preds_nocs = preds[..., 3]

                    loss_tr = torch.mean((preds_tr - targets_tr) ** 2, -1)
                    loss_up = (preds_up - targets_rot) ** 2
                    
                    targets_nocs = pc_canon[point_idx_all[:, 0]]
                    # loss_scale = torch.mean((preds_scale - targets_scale[None]) ** 2, -1)
                    loss_nocs = (preds_nocs - targets_nocs) ** 2
                    # print(targets_nocs.max(), targets_nocs.min(), preds_nocs.max(), preds_nocs.min())
                    
                    num_top = int(topk * loss_tr.shape[0])
                    loss_tr = torch.mean(torch.topk(loss_tr, num_top, largest=False, sorted=False)[0]) * 100
                    loss_up = torch.mean(torch.topk(loss_up, num_top, largest=False, sorted=False)[0])
                    # loss_scale = torch.mean(loss_scale)  # optimize scale loss for all pairs
                    # loss_nocs = torch.mean(loss_nocs)
                    loss_nocs = torch.mean(torch.topk(loss_nocs, num_top, largest=False, sorted=False)[0]) * 100
                    
                    loss = loss_up + loss_tr + loss_nocs # + loss_up_aux + loss_scale
                    loss.backward(retain_graph=False)
                    
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.)
                    opt.step()
                    
                    loss_meter.update(loss.item())
                    loss_tr_meter.update(loss_tr.item())
                    loss_up_meter.update(loss_up.item())
                    # loss_up_aux_meter.update(loss_up_aux.item())
                    # loss_scale_meter.update(loss_scale.item())
                    loss_nocs_meter.update(loss_nocs.item())
                        
                    n += 1
                    t.set_postfix(loss=loss_meter.avg, loss_tr=loss_tr_meter.avg, 
                            loss_up=loss_up_meter.avg, loss_nocs=loss_nocs_meter.avg)

            scheduler.step()
            
            # validation
            encoder.eval()
            # encoder.apply(apply_dropout)
            
            pcs, _, _, _, _, T_gt, up_gt = next(iter(t))
            T_gt = T_gt[0].cpu().numpy()
            up_gt = up_gt[0].cpu().numpy()
            pc = pcs[0].cpu().numpy()
            pc = pc.astype(np.float32)
            pcs = torch.from_numpy(pc[None]).cuda()
            
            point_idxs = np.random.randint(0, pc.shape[0], (num_pairs, 2))
            point_idx_more = torch.randint(0, pcs.shape[1], (point_idxs.shape[0], num_more), device='cuda')
            point_idx_all = torch.cat([torch.from_numpy(point_idxs).cuda(), point_idx_more], -1)
            inputs = torch.stack([pcs[0][point_idx_all[:, i]] - pcs[0][point_idx_all[:, j]] for (i, j) in combinations(np.arange(point_idx_all.shape[-1]), 2)], -2)
            
            ab = inputs[..., :3]
            mat = align_to(ab / (ab.norm(dim=-1, keepdim=True) + 1e-9), torch.tensor([1, 0, 0]).to(ab))
            inputs = (mat @ inputs[..., None])
            inputs = inputs.reshape(inputs.shape[0], -1)
            
            with torch.cuda.device(0):
                with torch.no_grad():
                    preds = encoder(inputs)[None]
                    preds_tr = preds[..., :2]
                    preds_up = preds[..., 2]
                    preds_nocs = preds[..., 3:6]
            
            
            # preds_tr = torch.from_numpy(generate_target_tr(pc, T_gt, point_idxs=point_idxs)[0]).cuda()[None]
            # import pdb; pdb.set_trace()
            grid_obj, candidates = validation(pc, preds_tr[0].cpu().numpy(), np.ones((point_idxs.shape[0],), dtype=np.float32), cfg.res, point_idxs, point_idxs.shape[0], num_rots, True)
            
            corners = np.stack([np.min(pc, 0), np.max(pc, 0)])
            T_est = candidates[-1]
            T_err_sp = np.linalg.norm(T_est - T_gt)
            print('pred sp translation error: ', T_err_sp)
            
            # preds_up = torch.from_numpy(generate_target_rot(pc, up_gt, point_idxs=point_idxs)[0]).cuda()[None]
            
            # sp rot voting
            final_directions = []
            for j, direction in enumerate([preds_up[:]]):
                with cp.cuda.Device(0):
                    candidates = cp.zeros((point_idxs.shape[0], num_rots, 3), cp.float32)

                    block_size = (point_idxs.shape[0] + 512 - 1) // 512
                    rot_voting_kernel(
                        (block_size, 1, 1),
                        (512, 1, 1),
                        (
                            cp.asarray(pc), cp.asarray(preds_tr[0].cpu().numpy()), cp.asarray(direction[0].cpu().numpy()), candidates, cp.asarray(point_idxs).astype(cp.int32), cp.asarray(corners[0]).astype(cp.float32), cp.float32(cfg.res), 
                            point_idxs.shape[0], num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2]
                        )
                    )
                candidates = candidates.get().reshape(-1, 3)
                
                with torch.no_grad():
                    sph_cp = torch.tensor(sphere_pts.T, dtype=torch.float32).cuda()
                    candidates = torch.from_numpy(candidates).cuda()
                    
                    counts = torch.zeros((sphere_pts.shape[0],), dtype=torch.float32, device='cuda')
                    for i in tqdm(range((candidates.shape[0] - 1) // bmm_size + 1)):
                        cos = candidates[i*bmm_size:(i+1)*bmm_size].mm(sph_cp)
                        counts += torch.sum((cos > np.cos(angle_tol * 2 / 180 * np.pi)).float(), 0)
                
                    for k, idx in enumerate(torch.topk(counts, 20)[1]):
                        d = sphere_pts[idx]
                        err = np.arccos(np.dot(d, up_gt)) / np.pi * 180
                        print('fine {} {} error: {:.05f} counts: {}'.format('up' if j == 0 else 'right', k, err, counts[idx].item()))
                best_dir = np.array(sphere_pts[np.argmax(counts.cpu().numpy())])
                final_directions.append(best_dir)
            
            up = final_directions[0]
            axis_err = np.arccos(np.dot(up, up_gt)) / np.pi * 180
            
            # nocs validation
            pc_vote = pc[point_idx_all[:, 0].cpu().numpy()]
            pred_coord = preds_nocs[0].cpu().numpy()
            
            up_final = None
            best_nocs_err = np.inf
            ups = [up, -up]
            for a in ups:
                rot = np.eye(3)
                
                b = np.array([0, -a[2], a[1]])
                b /= (np.linalg.norm(b) + 1e-9)
                
                # change this if choose another axis
                rot[:3, 0] = a
                rot[:3, 1] = b
                rot[:3, 2] = np.cross(a, b)
                pc_canon = (pc_vote - T_est) @ rot
                nocs_err = np.mean(np.abs(pred_coord[:, 0] - pc_canon[:, 0]))
                
                if nocs_err < best_nocs_err:
                    best_nocs_err = nocs_err
                    print('find better: ', best_nocs_err)
                    up_final = a
                
                if up_final is None:
                    up_final = a
            
            
            print('pred axis error (before nocs val): ', axis_err)
            print('pred axis error (after nocs val): ', np.arccos(np.dot(up_final, up_gt)) / np.pi * 180)
            
            visualize(vis, pc, np.stack(np.linspace(T_gt, T_gt + up * 0.1, 100)), win=6, opts=dict(markersize=3))
            # import pdb; pdb.set_trace()
            
        if not EVAL_ONLY:
            if loss_meter.avg < best_loss:
                best_loss = loss_meter.avg 
                torch.save(encoder.state_dict(), f'encoder_epochbest.pth')

if __name__ == '__main__':
    main()