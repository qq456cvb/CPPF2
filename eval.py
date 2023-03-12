from dataset import id2category
import numpy as np
from tqdm import tqdm
import cv2
import pickle
from pathlib import Path
from utils.util import backproject, downsample
import torch
from train import BeyondCPPF
from utils.util import compute_degree_cm_mAP
from omegaconf import OmegaConf


def main():
    torch.set_grad_enabled(False)
    path = 'checkpoints'
    log_dir = 'NOCS/nocs_output/real_test_20210511T2129'  # change it to your path
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    cfgs = {}
    ckpts = {}
    for i in range(1, 7):
        cat_name = id2category[i]
        cfg = OmegaConf.load(Path(path) / cat_name / '.hydra/config.yaml')
        ckpt_path = sorted((Path(path) / cat_name).glob('default/version*/checkpoints/last.ckpt'))[-1]
        ckpts[i] = BeyondCPPF.load_from_checkpoint(str(ckpt_path), cfg=cfg).eval().cuda()
        cfgs[i] = cfg
    
    result_pkl_list = Path(log_dir).rglob('results_*.pkl')
    result_pkl_list = sorted(result_pkl_list)

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
        
    for res in tqdm(final_results[:10]):
        # rgb = cv2.imread(res['image_path'] + '_color.png')[:, :, ::-1]
        depth = cv2.imread(res['image_path'] + '_depth.png', -1)
        bboxs = res['pred_bboxes']
        masks = res['pred_masks']
        RTs = res['pred_RTs']
        scales = res['pred_scales']
        cls_ids = res['pred_class_ids']
        
        for i, bbox in enumerate(bboxs):
            cls_id = cls_ids[i]
            cfg = cfgs[cls_id]
            model = ckpts[cls_id]

            mask = masks[:, :, i]
            pc, idxs = backproject(depth, intrinsics, mask)
            pc /= 1000
            pc[:, 0] = -pc[:, 0]
            pc[:, 1] = -pc[:, 1]
            pc = pc.astype(np.float32)

            indices = downsample(pc, cfg.res)
            pc = pc[indices]
            
            if pc.shape[0] > 50000 or ((pc.max(0) - pc.min(0)).max() / cfg.res) > 800:
                continue

            pred_rot, pred_center, pred_scale = model(pc)
            
            # a weird convension to NOCS evaluation
            pred_scale_norm = np.linalg.norm(pred_scale)
            RTs[i][:3, :3] = pred_rot * pred_scale_norm
            RTs[i][:3, -1] = pred_center
            scales[i] = pred_scale / pred_scale_norm
            
    aps = compute_degree_cm_mAP(final_results, ['BG', #0
                'bottle', #1
                'bowl', #2
                'camera', #3
                'can',  #4
                'laptop',#5
                'mug'#6
                ], 'plots',
                degree_thresholds = [5, 10, 15],#range(0, 61, 1), 
                shift_thresholds= [5, 10, 15], #np.linspace(0, 1, 31)*15, 
                iou_3d_thresholds=np.linspace(0, 1, 101),
                iou_pose_thres=0.1,
                use_matches_for_pose=True)
                
if __name__ == '__main__':
    main()