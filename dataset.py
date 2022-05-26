from pathlib import Path
import numpy as np
import torch
import os
from visdom import Visdom
import xmltodict
import logging
import open3d as o3d
from scipy.stats import special_ortho_group
import MinkowskiEngine as ME


def visualize(vis, *pcs, **opts):
    vis_pc = np.concatenate(pcs)
    vis_label = np.ones((sum([p.shape[0] for p in pcs])), np.int64)
    a = 0
    for i, pc in enumerate(pcs):
        vis_label[a:a+pc.shape[0]] = i + 1
        a += pc.shape[0]
    vis.scatter(vis_pc, vis_label, **opts)
    
    
def load_articulated_objects(root):
    anno_file = os.path.join(root, 'motion_unity.urdf')
    with open(anno_file, 'rb') as fp:
        data_dict = xmltodict.parse(fp.read())['robot']
    links = data_dict['link']
    if isinstance(links, dict):
        logging.warning(f'links should be list type, or there is only on part in object, root={root}')
    link2path = {}
    for link in links:
        name = link['@name']
        path = str(link['visual']['geometry']['mesh']['@filename'])[len('package://'):]
        path = os.path.join(root, path)
        link2path[name] = path
    # links_dict = {e['@name']: e for e in links}
    joints = data_dict['joint']
    # if only have on link, joints is dict, not list
    if isinstance(joints, dict):
        joints = [joints]
    joints_info = []
    for joint in joints:
        parent_name = joint['parent']['@link']
        child_name = joint['child']['@link']
        joint_info = {
            'parent': link2path[parent_name],
            'child': link2path[child_name],
            'origin': list(map(float, joint['origin']['@xyz'].split(' '))),
            'axis': list(map(float, joint['axis']['@xyz'].split(' '))),
            'lower': float(joint['limit']['@lower']),
            'upper': float(joint['limit']['@upper']),
        }
        joints_info.append(joint_info)
    return joints_info


def generate_target_tr(pc, o, subsample=200000, point_idxs=None):
    if point_idxs is None:
        point_idxs = np.random.randint(0, pc.shape[0], size=[subsample, 2])
                
    a = pc[point_idxs[:, 0]]
    b = pc[point_idxs[:, 1]]
    pdist = a - b
    pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
    proj_len = np.sum((a - o) * pdist_unit, -1)
    oc = a - o - proj_len[..., None] * pdist_unit
    dist2o = np.linalg.norm(oc, axis=-1)
    target_tr = np.stack([proj_len, dist2o], -1)
    
    return target_tr.astype(np.float32).reshape(-1, 2), point_idxs.astype(np.int64)


def generate_target_rot(pc, axis, subsample=200000, point_idxs=None):
    if point_idxs is None:
        point_idxs = np.random.randint(0, pc.shape[0], size=[subsample, 2])
    
    a = pc[point_idxs[:, 0]]
    b = pc[point_idxs[:, 1]]
    pdist = a - b
    pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
    target_rot = np.arccos(np.sum(pdist_unit * axis, -1))
    return target_rot.astype(np.float32).reshape(-1), point_idxs.astype(np.int64)

    
class ArticulatedDataset(torch.utils.data.Dataset):
    def __init__(self, root, cfg) -> None:
        super().__init__()
        self.joints = load_articulated_objects(root)
        self.mesh = o3d.io.read_triangle_mesh(os.path.join(root, 'eyeglasses1.obj'))
        self.vis = Visdom(port=21391)
        self.cfg = cfg
        
    def __getitem__(self, idx):
        order = [1, 2, 0]
        
        mesh = None
        for joint_info in self.joints:
            origin = np.array(joint_info['origin']).reshape(-1)[order]
            axis = np.array(joint_info['axis']).reshape(-1)[order]
            if mesh is None:
                mesh = o3d.io.read_triangle_mesh(joint_info['parent'])
            mesh = mesh + o3d.io.read_triangle_mesh(joint_info['child'])
            break
        
        # print(axis), original axis is hard to predict, while 1, 0, 0 and 0, 1, 0
        axis = np.array([1, 0, 0])
        pc = np.array(mesh.sample_points_uniformly(20000).points)
        pc = pc[ME.utils.sparse_quantize(pc, return_index=True, quantization_size=self.cfg.res)[1]]
        rand_rot = special_ortho_group.rvs(3)
        pc = pc @ rand_rot.T
        origin = rand_rot @ origin
        axis = rand_rot @ axis
        
        targets_tr, point_idxs = generate_target_tr(pc, origin)
        targets_rot, _ = generate_target_rot(pc, axis, point_idxs=point_idxs)
        
        pc_canon = (pc - origin) @ rand_rot
        return pc.astype(np.float32), pc_canon[..., 0].astype(np.float32), targets_tr.astype(np.float32), targets_rot.astype(np.float32), point_idxs.astype(int), origin.astype(np.float32), axis.astype(np.float32)
    
    def __len__(self):
        return 200
    

if __name__ == '__main__':
    cfg = lambda x: x
    cfg.res = 1e-3
    ds = ArticulatedDataset('/home/neil/ppf_matching/articulated/data/eyeglasses1', cfg)
    for d in ds:
        print(d[0].shape)
        import pdb; pdb.set_trace()