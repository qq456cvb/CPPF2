import os
import pickle as pkl
import json
import cv2
import numpy as np
import glob
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


def process_one_ann(args):
    ann_path, test_list, cur_path, class_names = args
    anns = pkl.load(open(ann_path, 'rb'))
    for ann in anns['annotations']:
        cls_n, seq_idx, obj_idx, frame_idx = ann['name'].split('/')
        if cls_n == 'cup':
            cls_n = 'mug'
        if (cls_n, seq_idx, obj_idx, str(int(frame_idx))) not in test_list:
            print((cls_n, seq_idx, obj_idx, str(int(frame_idx))), "not in test list")
            continue
        base_path = '{}/test_set/{}/{}/{}'.format(cur_path, cls_n, seq_idx, obj_idx)
        img_path = os.path.join(base_path, 'images/{}.jpg'.format(int(frame_idx)))
        if not os.path.isfile(img_path):
            img_path = os.path.join(base_path, 'rgbd/{}.jpg'.format(int(frame_idx)))
            print(img_path, "is not valid")
            continue

        output_path = os.path.join('{}/test_set/real275_fmt'.format(cur_path), '{}/{}/{}'.format(cls_n, seq_idx, obj_idx))
        os.makedirs(output_path, exist_ok=True)
        meta_path = os.path.join(base_path, 'metadata')

        # save rgbd mask meta camK
        output_img_id = '{:04d}'.format(int(frame_idx))

        if not (os.path.isfile(img_path) and os.path.isfile(img_path[:-4] + "-depth.png") and os.path.isfile(
                img_path[:-4] + "-mask.png")):
            print((cls_n, seq_idx, obj_idx, str(int(frame_idx))), "missing rgbd mask files")
            continue
        
        if not os.path.isfile(f"{output_path}/{output_img_id}_color.png"):
            os.symlink(img_path, f"{output_path}/{output_img_id}_color.png")
            os.symlink(img_path[:-4] + "-depth.png", f"{output_path}/{output_img_id}_depth.png")
            mask = cv2.imread(img_path[:-4] + "-mask.png", -1)
            cv2.imwrite(f"{output_path}/{output_img_id}_mask.png", (mask > 0).astype(np.uint8))

        with open(f"{output_path}/{output_img_id}_meta.txt", "w") as f:
            f.write(f"0 {category2id[cls_n]} {cls_n}\n")

        meta = json.load(open(meta_path, 'rb'))
        K = np.array(meta['K']).reshape(3, 3).T
        np.save(f"{output_path}/camK.npy", K)

        # save pose gt
        scale = ann['size']
        rot = ann['rotation']
        
        trans = ann['translation']
        RTs = np.eye(4)
        RTs[:3, :3] = rot
        RTs[:3, 3] = trans
        final_result = {
            'image_path': img_path,
            'gt_class_ids': [category2id[cls_n]],
            'gt_bboxes': [],
            'gt_RTs': [RTs],
            'gt_scales': [scale],
            'gt_handle_visibility': [1],
            # 'pred_class_ids': [],
            # 'pred_bboxes': [],
            # 'pred_RTs': [],
            # 'pred_scales': [],
            # 'pred_scores': [],
        }
        with open(f"{output_path}/{output_img_id}.pkl", "wb") as f:
            pkl.dump(final_result, f)


cur_path = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':
    class_names = ["mug"] #, "bottle", "bowl", "camera", "laptop"]
    todo_list = []
    for class_name in class_names:
        # load test list
        test_list = []
        if not os.path.isfile(f"test_set/test_list_{class_name}.txt"):
            continue
        with open(f"test_set/test_list_{class_name}.txt") as f:
            lines = [e.strip() for e in f.readlines()]
            for line in lines:
                line_strs = line.split("/")
                assert line_strs[-5] == class_name
                test_list.append((line_strs[-5], line_strs[-4], line_strs[-3], line_strs[-1][: -4]))

        # load annotation
        ann_paths = sorted(glob.glob('./test_set/pkl_annotations/{}/*.pkl'.format(class_name)))
        for ann_path in ann_paths:
            todo_list.append((ann_path, test_list, cur_path, class_names,))
    pool = multiprocessing.Pool(processes=10)
    list(tqdm(pool.imap_unordered(process_one_ann, todo_list), total=len(todo_list)))