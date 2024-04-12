import json
import os
import glob
import cv2
import numpy as np
import pickle as pkl
from tqdm import tqdm
import multiprocessing


category2id = {
    'bottle': 1,
    'bowl': 2,
    'camera': 3,
    'can': 4,
    'laptop': 5,
    'mug': 6
}

class2nocs = {
    0: 1, # bottle
    2: 4, # can
    3: 6, # mug
}

def process_one_seq(args):
    seq_path, class_obj_taxonomy = args
    with open(f"{seq_path}/scene_camera.json") as f:
        scene_camera = json.load(f)
    camK = np.eye(3)
    camK[0, 2] = scene_camera["rgb"]["cx"]
    camK[1, 2] = scene_camera["rgb"]["cy"]
    camK[0, 0] = scene_camera["rgb"]["fx"]
    camK[1, 1] = scene_camera["rgb"]["fy"]
    depth_scale = scene_camera["rgb"]["depth_scale"] * 1.0

    train_test_split = np.load(f"{seq_path}/train_test_split.npz")
    test_list = train_test_split["test_idxs"]

    with open(f"{seq_path}/rgb_scene_gt.json") as f:
        rgb_scene_gt = json.load(f)

    output_path = f"PhoCAL_release/real275_fmt/{os.path.basename(seq_path)}"
    os.makedirs(output_path, exist_ok=True)

    np.save(f"{output_path}/camK.npy", camK)

    for k, v in rgb_scene_gt.items():
        if int(k) not in test_list:
            continue
        img_id = "{:06d}".format(int(k))
        rgb_path = f"{seq_path}/rgb/{img_id}.png"
        depth_path = f"{seq_path}/depth/{img_id}.png"
        depth = cv2.imread(depth_path, -1) / depth_scale
        mask_path = f"{seq_path}/mask/{img_id}.png"
        mask = cv2.imread(mask_path, -1)
        # coord_map = cv2.imread(f"{seq_path}/nocs_map/{img_id}.png")[:, :, [2, 1, 0]]
        # coord_map = coord_map.astype(np.float32) / 255
        # coord_map[:, :, 2] = np.clip(1 - coord_map[:, :, 2], 0, 1) # ???

        # if not os.path.exists(rgb_path) or not os.path.exists(depth_path) or not os.path.exists(mask_path):
        os.symlink(rgb_path, f"{output_path}/{img_id}_color.png")
        os.symlink(depth_path, f"{output_path}/{img_id}_depth.png")
        os.symlink(mask_path, f"{output_path}/{img_id}_mask.png")
        meta_f = open(f"{output_path}/{img_id}_meta.txt", "w")

        final_result = {
            'image_path': f"{output_path}/{img_id}_color.png",
            'gt_class_ids': [],
            'gt_bboxes': [],
            'gt_RTs': [],
            'gt_scales': [],
            'gt_handle_visibility': [],
            'gt_mids': [],
            # 'pred_class_ids': [],
            # 'pred_bboxes': [],
            # 'pred_RTs': [],
            # 'pred_scales': [],
            # 'pred_scores': [],
        }
        for mid, rt_info in enumerate(v):
            cam_R_m2c = np.array(rt_info["cam_R_m2c"])
            cam_t_m2c = np.array(rt_info["cam_t_m2c"])
            class_id = rt_info["class_id"]
            inst_id = rt_info["inst_id"]
            
            if class_id not in [0, 2, 3]:
                continue
            
            nocs_class_id = class2nocs[class_id]
            # process coord map
            instance_mask = mask == (mid + 1)
            if np.sum((depth > 0) & instance_mask) == 0:
                continue
            
            RT = np.eye(4)
            RT[:3, 3] = cam_t_m2c
            RT[:3, :3] = cam_R_m2c.reshape(3, 3)
            
            scale = np.array(class_obj_taxonomy[str(class_id)]["scales"][str(inst_id)])
            if nocs_class_id in [1, 4, 6]:
                z = RT[:3, 2].copy()
                RT[:3, 2] = -RT[:3, 1]
                RT[:3, 1] = z
                scale = scale[[0, 2, 1]]
            
            meta_f.write(f"{mid} {nocs_class_id} {class_obj_taxonomy[str(class_id)]['objs'][str(inst_id)]}\n")
            final_result["gt_class_ids"].append(nocs_class_id)
            final_result['gt_mids'].append(mid + 1)
            final_result["gt_RTs"].append(RT)
            final_result["gt_scales"].append(scale)
            final_result["gt_handle_visibility"].append(1)

        meta_f.close()

        with open(f"{output_path}/{img_id}.pkl", "wb") as f:
            pkl.dump(final_result, f)


# PhoCAL_release/sequence_1
# depth  mask  nocs_map  polarization  pol_scene_gt.json  rgb  rgb_scene_gt.json  scene_camera.json  train_test_split.npz
cur_path = os.path.dirname(os.path.abspath(__file__))

root = f"{cur_path}/PhoCAL_release"
sequences = glob.glob(f"{root}/sequence_*")

class_obj_taxonomy = json.load(open(f"{root}/class_obj_taxonomy.json"))
class_names = []
for k, v in class_obj_taxonomy.items():
    class_names.append(v["class_name"])
# print(class_names)
# class_names = ['bottle', 'box', 'can', 'cup', 'remote', 'teapot', 'cutlery', 'glass']

with open(f"{root}/class_obj_taxonomy.json") as f:
    class_obj_taxonomy = json.load(f)

todo_list = []
for seq_path in sequences:
    todo_list.append((seq_path, class_obj_taxonomy,))
    
for args in tqdm(todo_list):
    process_one_seq(args)