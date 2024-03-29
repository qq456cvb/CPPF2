from glob import glob
import json
from pathlib import Path
import omegaconf
from utils.util import backproject, dilate_mask, fibonacci_sphere, real2prob, prob2real, calculate_2d_projections, draw, get_3d_bbox, process_data, transform_coordinates_3d, compute_degree_cm_mAP
import torch
from dataset import id2category
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import MinkowskiEngine as ME
import cv2
import pickle
import os
from src_shot.build import shot
from train_dino import vote_center, vote_rotation, generate_target_pairs
from train_dino import BeyondCPPF as BeyondCPPFDINO
from train_shot import BeyondCPPF as BeyondCPPFSHOT
from dataset import resize_crop
from visdom import Visdom
import torch_scatter
from dataset import DINOV2
import matplotlib
cm = matplotlib.colormaps['jet']
    

def visualize(vis, *pcs, **opts):
    vis_pc = np.concatenate(pcs)
    vis_label = np.ones((sum([p.shape[0] for p in pcs])), np.int64)
    a = 0
    for i, pc in enumerate(pcs):
        vis_label[a:a+pc.shape[0]] = i + 1
        a += pc.shape[0]
    vis.scatter(vis_pc, vis_label, **opts)


def get_topk_dir(pred, sphere_pts, bmm_size, angle_tol, wt=None, topk=1):
    sph_cp = torch.tensor(sphere_pts.T, dtype=torch.float32).cuda()
    counts = torch.zeros((sphere_pts.shape[0],), dtype=torch.float32, device='cuda')
    if wt is None:
        wt = torch.ones((pred.shape[0], 1)).to(pred)

    for i in range((pred.shape[0] - 1) // bmm_size + 1):
        cos = pred[i * bmm_size:(i + 1) * bmm_size].mm(sph_cp)
        counts += torch.sum((cos > np.cos(2 * angle_tol / 180 * np.pi)).float() / wt[i * bmm_size:(i + 1) * bmm_size], 0)

    # best_dir = np.array(sphere_pts[np.argmax(counts.cpu().numpy())])
    # return best_dir
    topk_idx = torch.topk(counts, topk)[1].cpu().numpy()
    topk_dir = np.array(sphere_pts[topk_idx])
    return topk_dir, counts.cpu().numpy()[topk_idx]

from PIL import Image
def main(
    angle_tol=1.,
    imp_wt_margin=0.01,
    backproj_ratio=.1,
    num_pairs=50000,
    num_rots=180,
    opt=True,
    debug=False,
    use_grounded_sam=False,
    geo_branch=True,
    visual_branch=True,
):
    desc_model = DINOV2().eval().cuda()
    
    torch.set_grad_enabled(False)
    dino_path = 'ckpts/dino'
    shot_path = 'ckpts/shot'
    
    print(angle_tol, imp_wt_margin, backproj_ratio, num_pairs, num_rots, opt, debug)

    if use_grounded_sam:
        log_dir = 'NOCS/nocs_output/real_test_groundedfastsam'
    else:
        log_dir = '/orion/u/yangyou/cppf/SAR-Net/results/NOCS/mrcnn_mask_results/real_test'  # refer SAR-Net for the mask, you can download from https://drive.google.com/file/d/1RwAbFWw2ITX9mXzLUEBjPy_g-MNdyHET/view
    whitelist = ['can', 'bowl', 'laptop', 'bottle', 'camera', 'mug']
    num_samples = int(4 * np.pi / (angle_tol / 180 * np.pi))
    sphere_pts = np.array(fibonacci_sphere(num_samples), dtype=np.float32)
    bmm_size = 100000
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

    dino_models = {}
    shot_models = {}
    cfgs = {}
    for i in range(1, 7):
        cat_name = id2category[i]
        if cat_name not in whitelist:
            continue
        root = f'{dino_path}/{cat_name}-num_more-3'
        cfg = omegaconf.OmegaConf.load(f"{root}/.hydra/config.yaml")
        dino_model = BeyondCPPFDINO.load_from_checkpoint(Path(root) / 'lightning_logs/version_0/checkpoints/last.ckpt', cfg=cfg).cuda().eval()
        dino_models[cat_name] = dino_model
        
        root = f'{shot_path}/{cat_name}-num_more-3'
        cfg = omegaconf.OmegaConf.load(f"{root}/.hydra/config.yaml")
        shot_model = BeyondCPPFSHOT.load_from_checkpoint(Path(root) / 'lightning_logs/version_0/checkpoints/last.ckpt', cfg=cfg).cuda().eval()
        shot_models[cat_name] = shot_model
        
        cfgs[cat_name] = cfg
        
    result_pkl_list = Path(log_dir).glob('results_*.pkl')
    result_pkl_list = sorted(result_pkl_list)[:]
    out_dir = 'NOCS/nocs_output/{}{}'.format(os.path.basename(__file__), '_opt' if opt else '_noopt')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    assert len(result_pkl_list)

    final_results = []
    for pkl_path in tqdm(result_pkl_list):
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
            # print(result)
            if not 'gt_handle_visibility' in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
                print('can\'t find gt_handle_visibility in the pkl.')
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(result['gt_handle_visibility'], result['gt_class_ids'])


        if type(result) is list:
            final_results += result
        elif type(result) is dict:
            final_results.append(result)
        else:
            assert False
    
    if debug:
        vis = Visdom()
        
    for res in tqdm(final_results[:]):
        image_path = res['image_path'].replace('data/real/test', 'NOCS/real_test')
        out_path = os.path.join(out_dir, '_'.join(image_path.split('/')[1:]) + '.pkl')
        # if os.path.exists(out_path):
        #     continue
        
        rgb = cv2.imread(image_path + '_color.png')[:, :, ::-1]
        depth = cv2.imread(image_path + '_depth.png', -1)
        bboxs = res['pred_bboxes']
        masks = res['pred_masks']
        
        res['pred_RTs'] = np.stack([np.eye(4) for _ in range(len(bboxs))])
        res['pred_scales'] = np.stack([np.ones((3,)) for _ in range(len(bboxs))])
        
        RTs = res['pred_RTs']
        scales = res['pred_scales']
        cls_ids = res['pred_class_ids']
        gt_RTs = res['gt_RTs']
        gt_scales = res['gt_scales']
        gt_cls_ids = res['gt_class_ids']
        draw_image_bbox = rgb.copy()
        for i, bbox in enumerate(bboxs):
            # gt_scale = gt_scales[gt_i]
            # print(R_est[:3, :3], r)
            # masks[:, :, i][bbox[0]:bbox[2], bbox[1]:bbox[3]] = True
            # continue
            # cv2.rectangle(draw_img, 
            #             (bbox[1], bbox[0]),
            #             (bbox[3], bbox[2]),
            #             (255, 0, 0), 2)
            
            cls_id = cls_ids[i]
            cat_name = id2category[cls_id]
            if cat_name not in whitelist:
                continue
            
            cat_name = id2category[cls_id]
            dino_model = dino_models[cat_name]
            shot_model = shot_models[cat_name]
            cfg = cfgs[cat_name]
            num_more = cfg.num_more
            mask = masks[:, :, i]
            
            # mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
            
            # vis.image(np.moveaxis(rgb, [0, 1, 2], [1, 2, 0]), win=1, opts=dict(width=640, height=480))
            # import pdb; pdb.set_trace()
            rgb_masked = np.zeros_like(rgb)
            rgb_masked[mask] = rgb[mask]
            rgb_local, transform = resize_crop(rgb_masked, bbox=Image.fromarray(rgb_masked).getbbox(), padding=0, out_size=256)
            # vis.image(np.moveaxis(rgb_masked, [0, 1, 2], [1, 2, 0]), win=1, opts=dict(width=640, height=480))

            # depth[depth > 0] += np.random.uniform(-2e-3, 2e-3, depth[depth > 0].shape)
            pc, idxs = backproject(depth / 1000., intrinsics, mask)
            idxs = np.stack(idxs, -1)  # K x 2
            pc[:, 0] = -pc[:, 0]
            pc[:, 1] = -pc[:, 1]
            pc = pc.astype(np.float32)
            # print(pc.shape, cfg.res)
            # vis.scatter(pc, win=8, opts=dict(markersize=3))
            indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=cfg.res)[1]
            pc = pc[indices]
            idxs = idxs[indices]
            if pc.shape[0] > 50000:
                sub_idx = np.random.randint(pc.shape[0], size=(50000,))
                pc = pc[sub_idx]
                idxs = idxs[sub_idx]
            # vis.scatter(pc, win=1, opts=dict(markersize=3))
            if ((pc.max(0) - pc.min(0)).max() / cfg.res) > 1000:
                continue
            
            kp = np.flip(idxs, -1)
            kp_local = (np.linalg.inv(transform) @ np.concatenate([kp, np.ones((kp.shape[0], 1))], -1).T).T[:, :2]
            desc = desc_model(torch.from_numpy(rgb_local).cuda().float().permute(2, 0, 1) / 255., torch.from_numpy(kp_local).float().cuda()).cpu().numpy()
            
            point_idxs_all = np.random.randint(0, pc.shape[0], (num_pairs, 2 + cfg.num_more))
            input_pairs = pc[point_idxs_all[:, :2]]
            
            shot_feat, normal = shot.compute(pc, cfg.res * 10, cfg.res * 10)
            shot_feat = shot_feat.reshape(-1, 352).astype(np.float32)
            # shot_feat = np.zeros((pc.shape[0], 352), dtype=np.float32)
            
            normal = normal.reshape(-1, 3).astype(np.float32)
            shot_feat[np.isnan(shot_feat)] = 0
            normal[np.isnan(normal)] = 0
            
            best_loss, best_idx = np.inf, 0
            for model_idx, model in enumerate([dino_model, shot_model]):
                if model_idx == 0:
                    pred_cls, pred_scales = model(torch.from_numpy(pc).float().cuda(), torch.from_numpy(desc).float().cuda(), torch.from_numpy(point_idxs_all).long().cuda())
                else:
                    pred_cls, pred_scales = model(torch.from_numpy(pc).float().cuda(), torch.from_numpy(point_idxs_all).long().cuda(),
                                                torch.from_numpy(shot_feat).float().cuda(), torch.from_numpy(normal).float().cuda())
                pred_pairs = pred_cls.reshape(pred_cls.shape[0], 2, 3, -1)
                num_bins = pred_cls.shape[-1]
                prob = torch.softmax(pred_cls, -1)
        
                pred_pairs = torch.multinomial(prob.reshape(np.product(pred_cls.shape[:-1]), -1), 1).float().reshape(-1, 2, 3)
                pred_pairs = (pred_pairs / (num_bins - 1) - 0.5)
                # pred_pairs = (prob2real(prob, 1., prob.shape[-1]) - 0.5).reshape(-1, 2, 3)  # this line gives worse results, just do multinomial by treating it as a sampling rather than expectation
                
                scale = torch.from_numpy(np.linalg.norm(input_pairs[:, 1] - input_pairs[:, 0], axis=-1)).float().cuda() \
                    / torch.clamp_min(torch.norm(pred_pairs[:, 1] - pred_pairs[:, 0], dim=-1), 1e-7)
                pred_pairs_scaled = pred_pairs * scale[:, None, None]
                
                targets_tr, targets_rot = generate_target_pairs(pred_pairs_scaled.cpu().numpy(), 
                                                                np.array(cfg.up),
                                                                np.array(cfg.front),
                                                                np.array(cfg.right))
                grid_obj, pred_trans = vote_center(torch.from_numpy(pc).float().cuda(), 
                    torch.from_numpy(targets_tr).float().cuda(), 
                    cfg.res, 
                    torch.from_numpy(point_idxs_all[:, :2]).long().cuda(),
                    num_rots=num_rots, 
                    vis=None)
                
                # if model_idx == 0:
                T_est = pred_trans
                
                # backvoting
                targets_tr_back, _ = generate_target_pairs(input_pairs, 
                                                                np.array(cfg.up),
                                                                np.array(cfg.front),
                                                                np.array(cfg.right),
                                                                T_est)
                back_errs = np.linalg.norm(targets_tr - targets_tr_back, axis=-1)
                pairs_mask = back_errs < np.percentile(back_errs, backproj_ratio * 100)
                # pairs_mask = back_errs < backproj_thres
                # print(pairs_mask.sum() / point_idxs_all.shape[0])
                point_idxs_pair_flattened = point_idxs_all[pairs_mask, :2].reshape(-1)
                unique_idx = np.unique(point_idxs_pair_flattened)
                pc_masked = pc[unique_idx]
                point_idxs_pair_flattened = torch.from_numpy(point_idxs_pair_flattened).cuda().long()
                imp_wt = torch_scatter.scatter_add(torch.ones_like(point_idxs_pair_flattened), 
                                                point_idxs_pair_flattened, dim=-1, dim_size=pc.shape[0]).cpu().numpy()

                
                point_idxs_all_filtered = point_idxs_all[pairs_mask]
                targets_rot = targets_rot[pairs_mask]
                scale = scale[pairs_mask]
                pred_scales = pred_scales[pairs_mask]
                
                imp_wt = imp_wt / imp_wt.max()
                imp_pair_wt = torch.from_numpy(imp_wt[point_idxs_all_filtered[:, :2]]).cuda().sum(-1) + imp_wt_margin  # N
                # imp_pair_wt.fill_(1.)
                preds_up, valid_mask = vote_rotation(torch.from_numpy(pc).float().cuda(),
                    torch.from_numpy(targets_rot[..., 0]).float().cuda(),
                    torch.from_numpy(point_idxs_all_filtered[:, :2]).long().cuda(),
                    num_rots)
                preds_up = preds_up.reshape(-1, 3)
                preds_ups, cnts = get_topk_dir(preds_up, sphere_pts, bmm_size, angle_tol, 
                                        imp_pair_wt[valid_mask, None].expand(-1, num_rots).reshape(-1, 1), topk=1)
                preds_up = preds_ups[0]
                
                preds_right, valid_mask = vote_rotation(torch.from_numpy(pc).float().cuda(),
                    torch.from_numpy(targets_rot[..., 2]).float().cuda(),
                    torch.from_numpy(point_idxs_all_filtered[:, :2]).long().cuda(),
                    num_rots)
                preds_right = preds_right.reshape(-1, 3)
                preds_rights, cnts = get_topk_dir(preds_right, sphere_pts, bmm_size, angle_tol,
                                        imp_pair_wt[valid_mask, None].expand(-1, num_rots).reshape(-1, 1), topk=1)
                preds_right = preds_rights[0]
                
                preds_right -= np.dot(preds_up, preds_right) * preds_up
                preds_right /= (np.linalg.norm(preds_right) + 1e-9)
                
                up_loc = np.where(cfg.up)[0][0]
                right_loc = np.where(cfg.right)[0][0]
                R_est = np.eye(3)
                R_est[:3, up_loc] = preds_up
                R_est[:3, right_loc] = preds_right
                
                gt_RT = gt_RTs[np.linalg.norm(gt_RTs[:, :3, -1] - T_est, axis=-1).argmin()]
                # pair_wt = F.normalize(1. / imp_pair_wt, p=1, dim=0)
                # pred_scale = torch.sum(pred_scales * pair_wt[:, None], 0).cpu().numpy()
                
                if model_idx == 0:
                    pred_scale = torch.median(pred_scales, 0)[0].cpu().numpy()
                    pred_scale_norm = np.linalg.norm(pred_scale)

                other_loc = list(set([0, 1, 2]) - set([up_loc, right_loc]))[0]
                R_est[:3, other_loc] = np.cross(R_est[:3, (other_loc + 1) % 3], R_est[:3, (other_loc + 2) % 3])
                
                # RTs[i][:3, :3] = R_est * pred_scale_norm
                # RTs[i][:3, -1] = T_est
                # scales[i] = pred_scale / pred_scale_norm
                
                if opt:
                    from lietorch import SO3
                    # only optimize the translation
                    with torch.enable_grad():
                        opt_trans = torch.nn.Parameter(torch.from_numpy(T_est).cuda().float(), requires_grad=True)
                        delta_rot = torch.tensor([0, 0, 0, 1.], requires_grad=True, device='cuda')
                        pc_cuda = torch.from_numpy(pc).float().cuda()
                        opt = optim.Adam([opt_trans, delta_rot], lr=1e-2)
                        # tq = tqdm(range(100))
                        for _ in range(100):
                            opt.zero_grad()
                            rot = SO3.InitFromVec(delta_rot).matrix()[:3, :3] @ torch.from_numpy(R_est).float().cuda()
                            pc_canon = (pc_cuda - opt_trans) @ rot
                            # import pdb; pdb.set_trace()
                            loss = torch.abs(pc_canon[point_idxs_all_filtered[:, :2]] - pred_pairs_scaled[pairs_mask])
                            if id2category[cls_id] in ['can', 'bottle', 'bowl']:
                                loss = loss[..., 1]
                            loss = loss.mean()
                            loss.backward()
                            # opt_trans.grad = opt_trans.grad * 1e-2
                            delta_rot.grad = delta_rot.grad / 180 * np.pi
                            opt.step()
                            # tq.set_description(f'loss: {loss.item():.4f}')
                            
                    
                    if cls_id in gt_cls_ids:
                        if id2category[cls_id] in ['can', 'bottle', 'bowl']:
                            rot_err = np.arccos(np.dot(R_est[:3, 1], gt_RT[:3, 1] / np.cbrt(np.linalg.det(gt_RT[:3, :3])))) / np.pi * 180
                        else:
                            rot_err = np.arccos((np.trace(R_est[:3, :3].T @ gt_RT[:3, :3] / np.cbrt(np.linalg.det(gt_RT[:3, :3]))) - 1.) / 2) / np.pi * 180
                        if debug:
                            print("rot err: ", rot_err)
                            print('tr err', np.linalg.norm(RTs[i][:3, -1] - gt_RT[:3, -1]))
                    
                    T_est = opt_trans.detach().cpu().numpy()
                    # pred_scale = opt_scale.detach().cpu().numpy() * pred_scale
                    # pred_scale_norm = np.linalg.norm(pred_scale)
                    R_est = (SO3.InitFromVec(delta_rot).matrix()[:3, :3] @ torch.from_numpy(R_est).float().cuda()).detach().cpu().numpy()
                    
                
                pc_canon = (pc - T_est) @ R_est / pred_scale_norm
                loss = np.abs(pc_canon[point_idxs_all_filtered[:, :2]] - pred_pairs[pairs_mask].cpu().numpy())
                if id2category[cls_id] in ['can', 'bottle', 'bowl']:
                    loss = loss[..., 1]
                loss = np.clip(loss, 0, 0.1)
                loss = loss.mean()
                
                
                
                if loss < best_loss and ((geo_branch and model_idx == 0) or (visual_branch and model_idx == 1)):
                    best_loss = loss
                    best_idx = model_idx
                    RTs[i][:3, :3] = R_est * pred_scale_norm
                    RTs[i][:3, -1] = T_est
                    scales[i] = pred_scale / pred_scale_norm
                
                if debug:
                    print(f'{model_idx} loss: {loss:.4f}')
                    if cls_id in gt_cls_ids:
                        if id2category[cls_id] in ['can', 'bottle', 'bowl']:
                            rot_err = np.arccos(np.dot(R_est[:3, 1], gt_RT[:3, 1] / np.cbrt(np.linalg.det(gt_RT[:3, :3])))) / np.pi * 180
                        else:
                            rot_err = np.arccos((np.trace(R_est[:3, :3].T @ gt_RT[:3, :3] / np.cbrt(np.linalg.det(gt_RT[:3, :3]))) - 1.) / 2) / np.pi * 180
                        print(f"{model_idx}: {id2category[cls_id]} rot err: ", rot_err)
                        print(f'{model_idx}: {id2category[cls_id]} tr err', np.linalg.norm(RTs[i][:3, -1] - gt_RT[:3, -1]))
        
            if debug:
                xyz_axis = 0.3 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
                transformed_axes = transform_coordinates_3d(xyz_axis, RTs[i])
                projected_axes = calculate_2d_projections(transformed_axes, intrinsics)

                bbox_3d = get_3d_bbox(scales[i, :], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RTs[i])
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                draw_image_bbox = draw(draw_image_bbox, projected_bbox, projected_axes, (255, 0, 0))
        
        if debug:
            vis.image(np.moveaxis(draw_image_bbox, [0, 1, 2], [1, 2, 0]), win=2)
            import pdb; pdb.set_trace()
        else:
            # pass
            pickle.dump(res, open(out_path, 'wb'))
    _ = compute_degree_cm_mAP(final_results[:], ['BG', #0
                'bottle', #1
                'bowl', #2
                'camera', #3
                'can',  #4
                'laptop',#5
                'mug'#6
                ], str(Path(out_dir, 'plots')),
                                                                degree_thresholds = [5, 10, 15],#range(0, 61, 1), 
                                                                shift_thresholds= [5, 10, 15], #np.linspace(0, 1, 31)*15, 
                                                                iou_3d_thresholds=np.linspace(0, 1, 101),
                                                                iou_pose_thres=0.1,
                                                                use_matches_for_pose=True)

from fire import Fire   
if __name__ == '__main__':
    Fire(main)
