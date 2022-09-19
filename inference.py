import torch
import numpy as np
from voting_torch import voting_kernels


def vote_directions(preds_right, preds_up, preds_front, point_idxs, num_rots, pc, sphere_pts, bmm_size, angle_tol):
    final_directions = []
    mean = torch.tensor([])
    std = torch.tensor([])
    for j, direction in enumerate([preds_right, preds_up, preds_front]):
        direction_mean = torch.abs(direction - np.pi / 2).mean() / np.pi * 180
        direction_std = torch.abs(direction - np.pi / 2).std() / np.pi * 180
        mean = torch.cat((mean, direction_mean[None].cpu()))
        std = torch.cat((std, direction_std[None].cpu()))

    abandon_idx = torch.argmin(mean)

    for j, direction in enumerate([preds_right, preds_up, preds_front]):
        if j == abandon_idx:
            final_directions.append(None)
        else:
            candidates = voting_kernels.vote_rotation(pc, direction, point_idxs, num_rots)
            candidates = candidates.reshape(-1, 3)
            sph_cp = sphere_pts.T
            counts = torch.zeros((sphere_pts.shape[0],), dtype=torch.float32, device='cuda')
            for i in range((candidates.shape[0] - 1) // bmm_size + 1):
                cos = candidates[i * bmm_size:(i + 1) * bmm_size].mm(sph_cp)
                counts += torch.sum((cos > np.cos(2 * angle_tol / 180 * np.pi)).float(), 0)

            best_dir = sphere_pts[counts.argmax()]
            # print(j, np.argmax(counts), counts.max(), counts.sum(), best_dir)
            final_directions.append([best_dir, counts.max()])
    return final_directions
    

def select_best_rotation(final_directions, cfg):
    right_loc = np.where(cfg.right)[0][0]
    up_loc = np.where(cfg.up)[0][0]
    front_loc = np.where(cfg.front)[0][0]

    def refine(a, b):
        b2 = b - torch.dot(a, b) * a
        b2 /= torch.linalg.norm(b2 + 1e-7)
        return b2

    cand = []
    for e in final_directions:
        if e is None:
            cand.append(None)
        else:
            cand.append(e[0])

    for i, direction in enumerate(final_directions):
        if direction is None:
            a = final_directions[(i + 1) % 3][0]
            b = final_directions[(i + 2) % 3][0]
            aw = final_directions[(i + 1) % 3][1]
            bw = final_directions[(i + 2) % 3][1]
            if aw > bw:
                b2 = refine(a, b)
                cand[i] = torch.cross(a, b2)
                cand[(i + 2) % 3] = b2
            else:
                a2 = refine(b, a)
                cand[i] = torch.cross(a2, b)
                cand[(i + 1) % 3] = a2
            break

    rot = torch.eye(3).cuda()
    rot[:, right_loc] = cand[0]
    rot[:, up_loc] = cand[1]
    rot[:, front_loc] = cand[2]
    return rot