import torch
from voting_torch import _C


def contiguous_tensor(tensor):
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


def vote_translation(pc, preds_tr, point_idxs, corners, res, n_rots):
    grid_x, grid_y, grid_z = ((corners[1] - corners[0]) / res).type(torch.int32) + 1
    grid_x, grid_y, grid_z = grid_x.item(), grid_y.item(), grid_z.item()
    # make sure grid size is even, to be compatible with the cc_torch
    pc, preds_tr, point_idxs, corners = [contiguous_tensor(e) for e in [pc, preds_tr, point_idxs, corners]]
    grid_obj = _C.vote_translation(pc, preds_tr, point_idxs.type(torch.int32), corners[0], res, n_rots, grid_x, grid_y, grid_z)
    if grid_x % 2:
        grid_x += 1
        grid_obj = torch.cat([grid_obj, torch.zeros(1, grid_y, grid_z).cuda()], dim=0)
    if grid_y % 2:
        grid_y += 1
        grid_obj = torch.cat([grid_obj, torch.zeros(grid_x, 1, grid_z).cuda()], dim=1)
    if grid_z % 2:
        grid_z += 1
        grid_obj = torch.cat([grid_obj, torch.zeros(grid_x, grid_y, 1).cuda()], dim=2)
    return grid_obj


def vote_rotation(pc, preds_rot, point_idxs, n_rots):
    pc, preds_rot, point_idxs = [contiguous_tensor(e) for e in [pc, preds_rot, point_idxs]]
    return _C.vote_rotation(pc, preds_rot, point_idxs.type(torch.int32), n_rots)


def backvote(pc, preds_tr, pred_center, point_idxs, corner, res, n_rots, grid_x, grid_y, grid_z, tol=3.0):
    pc, preds_tr, pred_center, point_idxs, corner = [contiguous_tensor(e) for e in [pc, preds_tr, pred_center, point_idxs, corner]]
    return _C.backvote(pc, preds_tr, pred_center, point_idxs.type(torch.int32), corner, res, n_rots, grid_x, grid_y, grid_z, tol)


if __name__ == '__main__':
    pc = torch.rand(1000, 3).cuda()
    sample_num = 100000
    preds_tr = torch.rand(sample_num, 2).cuda()
    preds_rot = torch.rand(sample_num).cuda()
    point_idxs = torch.randint(0, pc.shape[0], [sample_num, 2]).cuda()
    corners = torch.stack([pc.min(0)[0], pc.max(0)[0]])
    res = 2e-3
    n_rots = 120 
    pred_center = torch.zeros(3).cuda()
    import time
    t0 = time.time()
    grid_obj = vote_translation(pc, preds_tr, point_idxs, corners, res, n_rots)
    rot = vote_rotation(pc, preds_rot, point_idxs, n_rots)
    mask = backvote(pc, preds_tr, pred_center, point_idxs, corners[0], res, n_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2], tol=3.0)
    t1 = time.time()
    print(grid_obj[:10, :10, :10])
    print(mask[:10])
    print(grid_obj.shape)
    print(rot.shape)
    print(mask.dtype, mask.shape)
    print(t1 - t0)

