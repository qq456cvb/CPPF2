#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include "helper_math.cuh"


namespace voting
{
    __global__ void vote_translation(const float *points, const float *outputs, const float *probs,
            const int *point_idxs, float *grid_obj, const float *corner, const float res,
            int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z);
    __global__ void vote_rotation(const float *points, const float *preds_rot, float *outputs_up,
            const int *point_idxs, int n_ppfs, int n_rots);
    __global__ void backvote(const float *points, const float *outputs, bool *output_mask, const int *point_idxs,
            const float *corner, const float res, int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z,
            const float *gt_center, const float tol);
}


torch::Tensor vote_translation(const torch::Tensor &points, const torch::Tensor &outputs,
        const torch::Tensor &point_idxs, const torch::Tensor &corner, const float res, const int n_rots,
		const int grid_x, const int grid_y, const int grid_z);
torch::Tensor vote_rotation(const torch::Tensor &points, const torch::Tensor &preds_rot,
        const torch::Tensor &point_idxs, const int n_rots);
torch::Tensor backvote(const torch::Tensor &points, const torch::Tensor &outputs, const torch::Tensor &pred_center,
        const torch::Tensor &point_idxs, const torch::Tensor &corner, const float res, const int n_rots, 
		const int grid_x, const int grid_y, const int grid_z, const float tol);

