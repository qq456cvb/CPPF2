import os
from pathlib import Path
import numpy as np
from utils.util import backproject, map_sym, rotx, roty, downsample
import torch
import trimesh
from src_shot.build import shot

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

shapenet_obj_scales = {
    '02946921': [0.128, 0.18],
    '02876657': [0.2300, 0.4594],
    '02880940': [0.1851, 0.2381],
    '02942699': [0.1430, 0.2567],
    '03642806': [0.3862, 0.5353],
    '03797390': [0.1501, 0.1995]
}

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



class ShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, model_names):
        super().__init__()
        os.environ.update(
            OMP_NUM_THREADS = '1',
            OPENBLAS_NUM_THREADS = '1',
            NUMEXPR_NUM_THREADS = '1',
            MKL_NUM_THREADS = '1',
            PYOPENGL_PLATFORM = 'osmesa',
            PYOPENGL_FULL_LOGGING = '0'
        )
        self.cfg = cfg
        self.intrinsics = np.array([[591.0125, 0, 320], [0, 590.16775, 240], [0, 0, 1]])
        self.model_names = model_names
            
    def get_item_impl(self, model_name, cfg, intrinsics):
        from pyrender import DirectionalLight, SpotLight, Mesh, Scene,\
                     OffscreenRenderer, IntrinsicsCamera, RenderFlags
        r = OffscreenRenderer(viewport_width=640, viewport_height=480)
        shapenet_cls, mesh_name = model_name.split('/')
        path = f'{self.cfg.shapenet_root}/{shapenet_cls}/{mesh_name}/models/model_normalized.obj'
        mesh = trimesh.load(path)
        obj_scale = shapenet_obj_scales[f'{shapenet_cls}']
        
        mesh_pose = np.eye(4)
        y_angle = np.random.uniform(0, 2 * np.pi)
        x_angle = np.random.uniform(15 / 180 * np.pi, 75 / 180 * np.pi)
        yy_angle = np.random.uniform(-15 / 180 * np.pi, 15 / 180 * np.pi)
        # rotate to nocs coord
        flip2nocs = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        mesh_pose[:3, :3] = roty(yy_angle)[:3, :3] @ rotx(x_angle)[:3, :3] @ roty(y_angle)[:3, :3]
        # mesh_pose[:3, :3] = special_ortho_group.rvs(3)
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
        cam = IntrinsicsCamera(591.0125, 590.16775, 320, 240)
        
        direc_l = DirectionalLight(color=np.ones(3), intensity=np.random.uniform(5, 15))
        spot_l = SpotLight(color=np.ones(3), intensity=np.random.uniform(0, 10),
                        innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
        
        scene.add(cam, pose=cam_pose)
        scene.add(direc_l, pose=cam_pose)
        scene.add(spot_l, pose=cam_pose)
        
        depth = r.render(scene, flags=RenderFlags.DEPTH_ONLY)
        r.delete()
        
        mask = (depth > 0).astype(bool)
        idxs = np.where(mask)
        
        pc, _ = backproject(depth, intrinsics, mask)
        
        pc[:, 0] = -pc[:, 0]
        pc[:, 1] = -pc[:, 1]
        
        rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ mesh_pose[:3, :3] @ np.linalg.inv(flip2nocs)
        if self.cfg.up_sym:
            rot = map_sym(rot.T, np.where(self.cfg.up)[0][0]).T
        
        rt = np.eye(4)
        rt[:3, :3] = rot * scale
        rt[:3, -1] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ tr
        
        indices = downsample(np.ascontiguousarray(pc), cfg.res)
        pc = pc[indices]
        if pc.shape[0] < 100:
            return self.get_item_impl(self.model_names[np.random.randint(len(self))], self.cfg, self.intrinsics)
        
        pc_canon = pc - rt[:3, -1]
        pc_canon = (np.linalg.inv(rot) @ pc_canon.T).T
        
        bound = bounds[1] - bounds[0]
        bound[[0, 2]] = bound[[2, 0]]
        nocs = pc_canon / (bound.max() * scale)
        
        # vis.image(np.moveaxis(rgb, -1, 0), win=1)
        # depth /= depth.max()
        # vis.image(depth, win=2)
        
        # xyz_axis = 0.3 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
        # transformed_axes = transform_coordinates_3d(xyz_axis, rt)
        # projected_axes = calculate_2d_projections(transformed_axes, self.intrinsics)

        # bbox_3d = get_3d_bbox(bound, 0)
        # transformed_bbox_3d = transform_coordinates_3d(bbox_3d, rt)
        # projected_bbox = calculate_2d_projections(transformed_bbox_3d, self.intrinsics)
        # draw_image_bbox = draw(rgb.copy(), projected_bbox, projected_axes, (255, 0, 0))
        
        # vis.image(np.moveaxis(draw_image_bbox, [0, 1, 2], [1, 2, 0]), win=1)
        # vis.scatter(pc_canon, win=2, opts=dict(markersize=3))
        # scale_ratio = np.array([bound[1] / bound[0], bound[2] / bound[0]])
        
        targets_tr, targets_rot, point_idxs = generate_target_noaux(pc_canon, self.cfg.up, self.cfg.right, self.cfg.front, 100000)
        
        pc = pc.astype(np.float32)
        
        shot_feat = shot.compute(pc, self.cfg.res * 10, self.cfg.res * 10).reshape(-1, 352).astype(np.float32)
        shot_feat[np.isnan(shot_feat)] = 0
        
        normals = shot.estimate_normal(pc, self.cfg.res * 10).reshape(-1, 3).astype(np.float32)
        normals[~np.isfinite(normals)] = 0
        
        return {
            'pc': pc.astype(np.float32),
            'shot': shot_feat.astype(np.float32),
            'normal': normals.astype(np.float32),
            'nocs': nocs.astype(np.float32),
            'targets_scale': (bound * scale).astype(np.float32),
            'targets_tr': targets_tr.astype(np.float32),
            'targets_rot': targets_rot.astype(np.float32),
            'point_idxs': point_idxs.astype(int)
        }
        
    def __getitem__(self, idx):
        model_name = self.model_names[idx]
        return self.get_item_impl(model_name, self.cfg, self.intrinsics)
    
    def __len__(self):
        return len(self.model_names)