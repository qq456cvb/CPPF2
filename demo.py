from glob import glob
import json
from pathlib import Path
import omegaconf
from utils.util import downsample, backproject, dilate_mask, fibonacci_sphere, real2prob, prob2real, calculate_2d_projections, draw, get_3d_bbox, process_data, transform_coordinates_3d, compute_degree_cm_mAP
import torch
from dataset import id2category
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import cv2
import pickle
import os
try:
    from src_shot.build.Release import shot
except:
    from src_shot.build import shot
from train_dino import vote_center, vote_rotation, generate_target_pairs
from train_dino import BeyondCPPF as BeyondCPPFDINO
from train_shot import BeyondCPPF as BeyondCPPFSHOT
from dataset import resize_crop
from visdom import Visdom
import torch_scatter
from dataset import DINOV2
import matplotlib
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
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
import imageio
import tempfile

def extract_frames_from_video(video_path, temp_dir):
    """Extract frames from video file"""
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_idx += 1
    
    cap.release()
    return frame_paths

def create_gif_from_frames(frame_paths, output_path, fps=10):
    """Create GIF from list of frame paths"""
    images = []
    for frame_path in frame_paths:
        img = imageio.imread(frame_path)
        images.append(img)
    
    imageio.mimsave(output_path, images, fps=fps)
    print(f"GIF saved to: {output_path}")

def main(
    input_video=None,
    input_depth_dir=None,
    intrinsics_path=None,
    object_category='mug',
    output_gif='output.gif',
    frame_skip=1,
    fps=10,
    angle_tol=1.,
    imp_wt_margin=0.01,
    backproj_ratio=.1,
    num_pairs=50000,
    num_rots=180,
    opt=True,
    debug=False,
    geo_branch=True,
    visual_branch=True,
):
    # Validate input parameters
    if input_video is None:
        print("Error: Please provide input_video path")
        print("Usage: python demo.py --input_video path/to/video.mp4 --object_category mug")
        return
    
    if intrinsics_path is None:
        print("Error: Please provide intrinsics_path")
        print("Usage: python demo.py --input_video path/to/video.mp4 --intrinsics_path path/to/intrinsics.npy --object_category mug")
        return
    
    desc_model = DINOV2().eval().cuda()
    
    torch.set_grad_enabled(False)
    dino_path = 'ckpts/dino'
    shot_path = 'ckpts/shot'
    
    print(f"Processing video: {input_video}")
    print(f"Object category: {object_category}")
    print(f"Output GIF: {output_gif}")
    print(f"Parameters: angle_tol={angle_tol}, imp_wt_margin={imp_wt_margin}, backproj_ratio={backproj_ratio}")
    print(f"num_pairs={num_pairs}, num_rots={num_rots}, opt={opt}, debug={debug}")
    
    num_samples = int(4 * np.pi / (angle_tol / 180 * np.pi))
    sphere_pts = np.array(fibonacci_sphere(num_samples), dtype=np.float32)
    bmm_size = 100000
    
    # Load camera intrinsics
    try:
        intrinsics = np.load(intrinsics_path)
    except Exception as e:
        print(f"Error loading intrinsics from {intrinsics_path}: {e}")
        return

    cat_name = object_category
    root = f'{dino_path}/{cat_name}-num_more-3'
    cfg = omegaconf.OmegaConf.load(f"{root}/.hydra/config.yaml")
    dino_model = BeyondCPPFDINO.load_from_checkpoint(Path(root) / 'lightning_logs/version_0/checkpoints/last.ckpt', cfg=cfg).cuda().eval()
    
    root = f'{shot_path}/{cat_name}-num_more-3'
    cfg = omegaconf.OmegaConf.load(f"{root}/.hydra/config.yaml")
    shot_model = BeyondCPPFSHOT.load_from_checkpoint(Path(root) / 'lightning_logs/version_0/checkpoints/last.ckpt', cfg=cfg).cuda().eval()
    
    det_model = maskrcnn_resnet50_fpn(pretrained=True).eval().cuda()
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    prediction_frames = []
    
    try:
        # Extract frames from video
        print("Extracting frames from video...")
        frame_paths = extract_frames_from_video(input_video, temp_dir)
        
        # Create output directory
        os.makedirs('predictions', exist_ok=True)
        
        # Process frames with frame_skip
        selected_frames = frame_paths[::frame_skip]
        print(f"Processing {len(selected_frames)} frames (skipping every {frame_skip} frames)...")
        
        for frame_idx, frame_path in enumerate(tqdm(selected_frames)):
            rgb = cv2.imread(frame_path)[..., ::-1]
            
            # Try to load corresponding depth if depth directory is provided
            depth = None
            if input_depth_dir:
                depth_filename = f"frame_{frame_idx * frame_skip:06d}.npy"
                depth_path = os.path.join(input_depth_dir, depth_filename)
                if os.path.exists(depth_path):
                    depth = np.load(depth_path)
                else:
                    print(f"Warning: Depth file not found at {depth_path}, skipping this frame")
                    continue
            else:
                print("Warning: No depth directory provided, using dummy depth")
                # Create a dummy depth map - this won't work well for real pose estimation
                depth = np.ones((rgb.shape[0], rgb.shape[1])) * 1.0  # 1 meter depth
            
            draw_image_bbox = rgb.copy()
            
            # Object detection and masking
            image_tensor = F.to_tensor(Image.fromarray(rgb)).unsqueeze(0).cuda()
            prediction = det_model(image_tensor)
            
            # Get the class ID for the target object (this is a simplification)
            # In practice, you might want to use a mapping from category name to class ID
            target_class_mapping = {
                'mug': 47,  # COCO class ID for cup
                'bottle': 44,  # COCO class ID for bottle
                'bowl': 51,  # COCO class ID for bowl
                # Add more mappings as needed
            }
            
            target_class_id = target_class_mapping.get(object_category, 47)  # Default to mug
            
            mask = None
            for i, class_id in enumerate(prediction[0]['labels']):
                if class_id == target_class_id:
                    mask = prediction[0]['masks'][i, 0]
                    mask = (mask > 0.2).cpu().numpy()  # Convert to boolean format
                    break
            
            if mask is None:
                print(f"Warning: No {object_category} detected in frame {frame_idx}, skipping")
                continue
            
            mask_show = rgb.copy()
            mask_show[~mask] = 0
        
            # mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        
            # vis.image(np.moveaxis(rgb, [0, 1, 2], [1, 2, 0]), win=1, opts=dict(width=640, height=480))
            # import pdb; pdb.set_trace()
            rgb_masked = np.zeros_like(rgb)
            rgb_masked[mask] = rgb[mask]
            rgb_local, transform = resize_crop(rgb_masked, bbox=Image.fromarray(rgb_masked).getbbox(), padding=0, out_size=256)
            # vis.image(np.moveaxis(rgb_masked, [0, 1, 2], [1, 2, 0]), win=1, opts=dict(width=640, height=480))

            # depth[depth > 0] += np.random.uniform(-2e-3, 2e-3, depth[depth > 0].shape)
            pc, idxs = backproject(depth, intrinsics, mask)
            idxs = np.stack(idxs, -1)  # K x 2
            pc[:, 0] = -pc[:, 0]
            pc[:, 1] = -pc[:, 1]
            pc = pc.astype(np.float32)
            # print(pc.shape, cfg.res)
            # vis.scatter(pc, win=8, opts=dict(markersize=3))
            indices = downsample(pc, cfg.res)
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
            best_RT = np.eye(4)
            best_scale = np.zeros((3,))
            pred_scale_norm = 1.0  # Initialize with default value
            pred_scale = np.ones((3,))  # Initialize with default value
            
            for model_idx, model in enumerate([dino_model, shot_model]):
                if model_idx == 0:
                    pred_cls, pred_scales = model(torch.from_numpy(pc).float().cuda(), torch.from_numpy(desc).float().cuda(), torch.from_numpy(point_idxs_all).long().cuda())
                else:
                    pred_cls, pred_scales = model(torch.from_numpy(pc).float().cuda(), torch.from_numpy(point_idxs_all).long().cuda(),
                                                torch.from_numpy(shot_feat).float().cuda(), torch.from_numpy(normal).float().cuda())
                pred_pairs = pred_cls.reshape(pred_cls.shape[0], 2, 3, -1)
                num_bins = pred_cls.shape[-1]
                prob = torch.softmax(pred_cls, -1)
        
                pred_pairs = torch.multinomial(prob.reshape(int(np.prod(pred_cls.shape[:-1])), -1), 1).float().reshape(-1, 2, 3)
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
                            loss = loss.mean()
                            loss.backward()
                            # opt_trans.grad = opt_trans.grad * 1e-2
                            if delta_rot.grad is not None:
                                delta_rot.grad = delta_rot.grad / 180 * np.pi
                            opt.step()
                            # tq.set_description(f'loss: {loss.item():.4f}')
                    
                    T_est = opt_trans.detach().cpu().numpy()
                    R_est = (SO3.InitFromVec(delta_rot).matrix()[:3, :3] @ torch.from_numpy(R_est).float().cuda()).detach().cpu().numpy()
                    
                
                pc_canon = (pc - T_est) @ R_est / pred_scale_norm
                loss = np.abs(pc_canon[point_idxs_all_filtered[:, :2]] - pred_pairs[pairs_mask].cpu().numpy())
                loss = np.clip(loss, 0, 0.1)
                loss = loss.mean()
                
                if loss < best_loss and ((geo_branch and model_idx == 0) or (visual_branch and model_idx == 1)):
                    best_loss = loss
                    best_idx = model_idx
                    best_RT[:3, :3] = R_est * pred_scale_norm
                    best_RT[:3, -1] = T_est
                    best_scale = pred_scale / pred_scale_norm
    
            xyz_axis = 0.3 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            transformed_axes = transform_coordinates_3d(xyz_axis, best_RT)
            projected_axes = calculate_2d_projections(transformed_axes, intrinsics)

            bbox_3d = get_3d_bbox(best_scale, 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, best_RT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            draw_image_bbox = draw(draw_image_bbox, projected_bbox, projected_axes, (255, 0, 0))
        
            output_frame_path = f'predictions/frame_{frame_idx:06d}.png'
            cv2.imwrite(output_frame_path, draw_image_bbox[..., ::-1])
            prediction_frames.append(output_frame_path)
            
            # Optional: show frames during processing (disabled by default)
            if debug:
                cv2.imshow('prediction', draw_image_bbox[..., ::-1])
                cv2.waitKey(30)  # Show for 30ms
        
        if debug:
            cv2.destroyAllWindows()
            
        # Create GIF from prediction frames
        print("Creating GIF from prediction frames...")
        create_gif_from_frames(prediction_frames, output_gif, fps=fps)
        
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Cleaned up temporary directory: {temp_dir}")

from fire import Fire   
if __name__ == '__main__':
    Fire(main)
