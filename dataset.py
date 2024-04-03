import os
from pathlib import Path
import numpy as np
from utils.util import backproject, real2prob, map_sym, map_sym_discrete
import hydra
import torch
import omegaconf
import trimesh
from utils.util import backproject, calculate_2d_projections, fibonacci_sphere, get_3d_bbox, transform_coordinates_3d, draw, Unsharpen, DebayerArtefacts
import albumentations as A
import zmq
from src_shot.build import shot
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from itertools import combinations
from scipy.stats import special_ortho_group
import open3d as o3d
from utils.util import downsample

import logging
logger = logging.getLogger("OpenGL.arrays.arraydatatype")
logger.setLevel(logging.ERROR)
logger = logging.getLogger('OpenGL.acceleratesupport')
logger.setLevel(logging.ERROR)


category2id = {
    'bottle': 1,
    'bowl': 2,
    'camera': 3,
    'can': 4,
    'laptop': 5,
    'mug': 6
}
id2category = dict([(v, k) for (k, v) in category2id.items()])


def interpolate_features(descriptors, pts, strides=8, normalize=True):
    # Normalize keypoints to [-1, 1]
    h, w = descriptors.shape[-2], descriptors.shape[-1]
    
    keypoints = pts.clone()
    # convert keypoint location to pixel center
    keypoints[..., 0] = ((keypoints[..., 0] + 0.5) / w / strides) * 2 - 1  # x coordinates
    keypoints[..., 1] = ((keypoints[..., 1] + 0.5) / h / strides) * 2 - 1  # y coordinates
    
    # Expand dimensions for grid sampling
    keypoints = keypoints.unsqueeze(-3)  # Shape becomes [batch_size, 1, num_keypoints, 2]
    
    # Interpolate using bilinear sampling
    interpolated_features = F.grid_sample(descriptors, keypoints, align_corners=False)
    
    # interpolated_features will have shape [batch_size, channels, 1, num_keypoints]
    # You might want to squeeze or reshape as necessary.
    interpolated_features = interpolated_features.squeeze(-2)
    
    return F.normalize(interpolated_features, dim=1) if normalize else interpolated_features
    
    
class DINOV2(nn.Module):
    def __init__(self, stride=4):
        super().__init__()
        self.dinov2_vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').eval()
        self.transform = None
        self.stride = stride
    
    # rgb: 3, h, w
    def forward(self, rgb, pts):
        if self.transform is None:
            self.patch_h, self.patch_w = rgb.shape[-2] // self.stride, rgb.shape[-1] // self.stride
            self.transform = T.Compose([
                T.Resize((self.patch_h * 14, self.patch_w * 14)),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        result = self.dinov2_vit.forward_features(self.transform(rgb.unsqueeze(0)))
        raw_descs = result['x_norm_patchtokens'].reshape(1, self.patch_h, self.patch_w, -1).permute(0, 3, 1, 2)
        features = interpolate_features(raw_descs, pts[None], strides=self.stride, normalize=True)[0].T
        return features
    
    

def rotz(a):
    return np.array([[np.cos(a), np.sin(a), 0, 0],
                        [-np.sin(a), np.cos(a), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    

def roty(a):
    return np.array([[np.cos(a), 0, -np.sin(a), 0],
                        [0, 1, 0, 0],
                        [np.sin(a), 0, np.cos(a), 0],
                        [0, 0, 0, 1]])
    
def rotx(a):
    return np.array([[1, 0, 0, 0],
                        [0, np.cos(a), -np.sin(a), 0],
                        [0, np.sin(a), np.cos(a), 0],
                        [0, 0, 0, 1]])


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def downsample(pc, res):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    _, _, idxs = pcd.voxel_down_sample_and_trace(res, pcd.get_min_bound(), pcd.get_max_bound())
    res = []
    for idx in idxs:
        res.append(np.random.choice(np.array(idx)))
    return np.array(res)


def generate_target_pairs(point_pairs, up, right, front, center=np.zeros((3,))):
    a = point_pairs[:, 0]
    b = point_pairs[:, 1]
    pdist = a - b
    pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
    proj_len = np.sum((a - center) * pdist_unit, -1)
    oc = (a - center) - proj_len[..., None] * pdist_unit
    dist2o = np.linalg.norm(oc, axis=-1)
    # print(proj_len.shape, dist2o.shape)
    # print(proj_len.min(), proj_len.max())
    target_tr = np.stack([proj_len, dist2o], -1)
    
    up_cos = np.arccos(np.sum(pdist_unit * up, -1))
    right_cos = np.arccos(np.sum(pdist_unit * right, -1))
    front_cos = np.arccos(np.sum(pdist_unit * front, -1))
    target_rot = np.stack([up_cos, right_cos, front_cos], -1)
    
    return target_tr.astype(np.float32).reshape(-1, 2), target_rot.astype(np.float32).reshape(-1, 3)

def generate_target_noaux(pc, up, right, front, subsample=200000, point_idxs=None):
    if point_idxs is None:
        if subsample is None:
            xv, yv = np.meshgrid(np.arange(pc.shape[1]), np.arange(pc.shape[1]))
            point_idxs = np.stack([yv, xv], -1).reshape(-1, 2)
        else:
            point_idxs = np.random.randint(0, pc.shape[0], size=[subsample, 2])
                
    a = pc[point_idxs[:, 0]]
    b = pc[point_idxs[:, 1]]
    pdist = a - b
    pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
    proj_len = np.sum(a * pdist_unit, -1)
    oc = a - proj_len[..., None] * pdist_unit
    dist2o = np.linalg.norm(oc, axis=-1)
    # print(proj_len.shape, dist2o.shape)
    # print(proj_len.min(), proj_len.max())
    target_tr = np.stack([proj_len, dist2o], -1)
    
    up_cos = np.arccos(np.sum(pdist_unit * up, -1))
    right_cos = np.arccos(np.sum(pdist_unit * right, -1))
    front_cos = np.arccos(np.sum(pdist_unit * front, -1))
    target_rot = np.stack([up_cos, right_cos, front_cos], -1)
    
    return target_tr.astype(np.float32).reshape(-1, 2), target_rot.astype(np.float32).reshape(-1, 3), point_idxs.astype(np.int64)



shapenet_obj_scales = {
    '02946921': [0.128, 0.18],
    '02876657': [0.16, 0.25],
    '02880940': [0.1851, 0.26],
    '02942699': [0.1430, 0.28],
    '03642806': [0.3862, 0.58],
    '03797390': [0.1501, 0.1995]
}
# vis = visdom.Visdom()

from icecream import ic

class ShapeNetDirectDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, full_rot=False):
        super().__init__()
        os.environ.update(
            # OMP_NUM_THREADS = '1',
            # OPENBLAS_NUM_THREADS = '1',
            # NUMEXPR_NUM_THREADS = '1',
            # MKL_NUM_THREADS = '1',
            PYOPENGL_PLATFORM = 'egl',
            PYOPENGL_FULL_LOGGING = '0'
        )
        self.cfg = cfg
        self.intrinsics = np.array([[591.0125, 0, 320], [0, 590.16775, 240], [0, 0, 1]])
        
        model_names = open(hydra.utils.to_absolute_path('data/shapenet_train.txt')).read().splitlines() + open(hydra.utils.to_absolute_path('data/shapenet_val.txt')).read().splitlines()
        model_names = [line.split()[1] for line in model_names if int(line.split()[0]) == cfg.category]
    
        self.model_names = []
        # blacklists = open(hydra.utils.to_absolute_path('data/blacklists.txt')).read().splitlines()
        for name in model_names:
            # if name not in blacklists:
            self.model_names.append(name)
        self.r = None
        self.full_rot = full_rot
            
    def get_item_impl(self, model_name, cfg, intrinsics):
        import OpenGL
        OpenGL.FULL_LOGGING = False
        OpenGL.ERROR_LOGGING = False
        from pyrender import IntrinsicsCamera,\
                     DirectionalLight, SpotLight, Mesh, Scene,\
                     OffscreenRenderer, RenderFlags
        if self.r is None:
            self.r = OffscreenRenderer(viewport_width=640, viewport_height=480)
        shapenet_cls, mesh_name = model_name.split('/')
        path = f'/orion/group/ShapeNetCore.v2/{shapenet_cls}/{mesh_name}/models/model_normalized.obj'
        mesh = trimesh.load(path)
        obj_scale = shapenet_obj_scales[f'{shapenet_cls}']
        mesh_pose = np.eye(4)
        y_angle = np.random.uniform(0, 2 * np.pi)
        x_angle = np.random.uniform(10 / 180 * np.pi, 80 / 180 * np.pi)
        yy_angle = np.random.uniform(-20 / 180 * np.pi, 20 / 180 * np.pi)
        # rotate to nocs coord
        flip2nocs = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        
        if self.full_rot:
            mesh_pose[:3, :3] = special_ortho_group.rvs(3)
        else:
            mesh_pose[:3, :3] = roty(yy_angle)[:3, :3] @ rotx(x_angle)[:3, :3] @ roty(y_angle)[:3, :3]
        tr = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3), -np.random.uniform(0.6, 2.0)])
        mesh_pose[:3, -1] = tr
        
        bounds = mesh.bounds
        trans_mat = np.eye(4)
        trans_mat[:3, -1] = -(bounds[1] + bounds[0]) / 2
        scale_mat = np.eye(4)
        scale = np.random.uniform(obj_scale[0], obj_scale[1])
        scale_mat[:3, :3] *= scale
        mesh.apply_transform(mesh_pose @ scale_mat @ trans_mat)
        if isinstance(mesh, trimesh.Scene):
            scene = Scene.from_trimesh_scene(mesh)
            scene.bg_color = np.zeros((3,))
        else:
            scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]), bg_color=np.zeros((3,)))
            scene.add(Mesh.from_trimesh(mesh), pose=np.eye(4))

        cam_pose = np.eye(4)
        # cam = PinholeCamera(591.0125, 590.16775, 640, 480)
        cam = IntrinsicsCamera(591.0125, 590.16775, 320, 240)
        
        direc_l = DirectionalLight(color=np.ones(3), intensity=np.random.uniform(5, 15))
        spot_l = SpotLight(color=np.ones(3), intensity=np.random.uniform(0, 10),
                        innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
        
        scene.add(cam, pose=cam_pose)
        scene.add(direc_l, pose=cam_pose)
        scene.add(spot_l, pose=cam_pose)
        
        rgb, depth = self.r.render(scene)
        
        mask = (depth > 0).astype(bool)
        # depth[mask] += np.random.uniform(-4e-3, 4e-3, depth[mask].shape)
        pc, idxs = backproject(depth, intrinsics, mask)
        idxs = np.stack(idxs, -1)  # K x 2
        pc[:, 0] = -pc[:, 0]
        pc[:, 1] = -pc[:, 1]
        
        rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ mesh_pose[:3, :3] @ np.linalg.inv(flip2nocs)  # need to transform back into opencv coord
        if self.cfg.category in [1, 2, 4]:
            rot = map_sym(rot.T, np.where(self.cfg.up)[0][0]).T
            
        trans = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ tr
        bound = bounds[1] - bounds[0]
        bound[[0, 2]] = bound[[2, 0]]
        
        indices = downsample(pc, self.cfg.res)
        pc = pc[indices]
        idxs = idxs[indices]
        if pc.shape[0] < 100:
            return self.get_item_impl(self.model_names[np.random.randint(len(self))], self.cfg, self.intrinsics)
        
        shot_feat, normal = shot.compute(pc, self.cfg.res * 10, self.cfg.res * 10)
        shot_feat = shot_feat.reshape(-1, 352).astype(np.float32)
        normal = normal.reshape(-1, 3).astype(np.float32)
        shot_feat[np.isnan(shot_feat)] = 0
        normal[np.isnan(normal)] = 0

        point_idxs_all = np.random.randint(0, pc.shape[0], (10000, 2 + self.cfg.num_more))
        bound = bound * scale
        scale = bound.max()
        
        pc_canon = (pc - trans) @ rot
        pc_canon /= scale
        # vis.scatter(pc_canon, win=2, opts=dict(markersize=3))
        # vis.image(np.moveaxis(rgb, -1, 0), win=1)
        # vis.scatter(pc, win=1, opts=dict(markersize=3))
        # print(bound * scale)
        quat = R.from_matrix(rot).as_quat()[[3, 0, 1, 2]]  # convert to wxyz
        # pc = pc_canon
        # trans[:] = 0
        # quat[:] = 0
        # quat[0] = 1
        return {
            'pc': pc.astype(np.float32),
            'pc_canon': pc_canon.astype(np.float32),
            'trans': trans.astype(np.float32),
            'quat': quat.astype(np.float32),
            'bound': bound.astype(np.float32),
            'scale': scale.astype(np.float32),
            'point_idxs_all': point_idxs_all.astype(int),
            'rgb': rgb.astype(np.uint8),
            'depth': depth.astype(np.float32),
            'idxs': idxs.astype(np.int64),
            'shot': shot_feat,
            'normal': normal
        }
        
    def __getitem__(self, idx):
        model_name = self.model_names[idx]
        return self.get_item_impl(model_name, self.cfg, self.intrinsics)
    
    def __len__(self):
        return len(self.model_names)

from torchvision.transforms import functional
def resize_crop(img, padding=0.2, out_size=224, bbox=None):
    # return np.array(img), np.eye(3)
    img = Image.fromarray(img)
    if bbox is None:
        bbox = img.getbbox()
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    size = max(height, width) * (1 + padding)
    center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
    bbox_enlarged = center[0] - size / 2, center[1] - size / 2, \
        center[0] + size / 2, center[1] + size / 2
    img = functional.resize(functional.crop(img, bbox_enlarged[1], bbox_enlarged[0], size, size), (out_size, out_size))
    transform = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1.]])  \
        @ np.array([[size / out_size, 0, 0], [0, size / out_size, 0], [0, 0, 1]]) \
        @ np.array([[1, 0, -out_size / 2], [0, 1, -out_size / 2], [0, 0, 1.]])
    return np.array(img), transform



class ShapeNetExportDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, full_rot=False):
        super().__init__()
        self.cfg = cfg
        self.category = cfg.category
        self.root = hydra.utils.to_absolute_path('data/category_training_data{}/{}'.format('_full_rot' if full_rot else '', cfg.category))
        model_names = open(hydra.utils.to_absolute_path('data/shapenet_train.txt')).read().splitlines() + open(hydra.utils.to_absolute_path('data/shapenet_val.txt')).read().splitlines()
        self.model_names = [line.split()[1] for line in model_names if int(line.split()[0]) == cfg.category]
        self.blacklists = open(hydra.utils.to_absolute_path('data/blacklists.txt')).read().splitlines()
        self.blacklist_idxs = [self.model_names.index(name) for name in self.blacklists if name in self.model_names]
        self.candidate_idxs = list(set(range(len(self.model_names))) - set(self.blacklist_idxs))
        # if cfg.category == 5:
        #     self.candidate_idxs = [282, 10, 449, 276, 92, 366, 387, 229, 409, 309, 201, 71]
        assert(len(self.model_names) == len(list(Path(self.root).glob('*.pkl'))) // 100)
            
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of bounds")
        idx = np.random.randint(100) * np.random.choice(self.candidate_idxs)
        data = pickle.load(open(os.path.join(self.root, '{:06d}.pkl'.format(idx)), 'rb'))
        return data

    def __len__(self):
        return 200
        
        
from tqdm import tqdm
from PIL import Image
import pickle

def dump_data(full_rot=False):
    cfg = omegaconf.OmegaConf.load('config/config.yaml')
    for cat in range(1, 7):
        cfg.category = cat
        ds = ShapeNetDirectDataset(cfg, full_rot=full_rot)
        desc_model = DINOV2().eval().cuda()
        torch.set_grad_enabled(False)
        cnt = 0
        tq = tqdm(total=len(ds) * 100)
        if not Path('data/category_training_data{}/{}'.format('_full_rot' if full_rot else None, cfg.category)).exists():
            os.makedirs('data/category_training_data{}/{}'.format('_full_rot' if full_rot else None, cfg.category))
        for _ in range(100):
            for d in ds:
                rgb = d['rgb']
                depth = d['depth']
                idxs = d['idxs']
                pc = d['pc']
                pc_canon = d['pc_canon']
                bound = d['bound']
                # shots = d['shot']
                # normals = d['normal']
                
                # # rgb[idx] = pc
                rgb_local, transform = resize_crop(rgb, bbox=Image.fromarray(depth).getbbox(), padding=0.5, out_size=256)
                
                # random choose 100 points
                sub_idx = np.random.choice(np.arange(idxs.shape[0]), 100)
                idxs = idxs[sub_idx]
                kp = np.flip(idxs, -1)
                kp_local = (np.linalg.inv(transform) @ np.concatenate([kp, np.ones((kp.shape[0], 1))], -1).T).T[:, :2]
                
                desc = desc_model(torch.from_numpy(rgb_local).cuda().float().permute(2, 0, 1) / 255., torch.from_numpy(kp_local).float().cuda()).cpu().numpy()
                
                pickle.dump({
                    'pc': pc[sub_idx],
                    'pc_canon': pc_canon[sub_idx],
                    'desc': desc,
                    'bound': bound,
                    # 'shot': shots[sub_idx],
                    # 'normal': normals[sub_idx],
                }, open('data/category_training_data{}/{}/{:06d}.pkl'.format('_full_rot' if full_rot else None, cfg.category, cnt), 'wb'))
                cnt += 1
                tq.update(1)
                import pdb; pdb.set_trace()
                

if __name__ == '__main__':
    dump_data(False)