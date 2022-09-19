#include "vote_kernels.h"

#define THREAD_NUM 512

namespace voting
{
    __global__ void vote_translation(const float *points, const float *outputs, const float *probs,
            const int *point_idxs, float *grid_obj, const float *corner, const float res,
            int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float proj_len = outputs[idx * 2];
            float odist = outputs[idx * 2 + 1];
            if (odist < res) return;
            int a_idx = point_idxs[idx * 2];
            int b_idx = point_idxs[idx * 2 + 1];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 ab = a - b;
            if (length(ab) < 1e-7) return;
            ab /= (length(ab) + 1e-7);
            float3 c = a - ab * proj_len;

            // float prob = max(probs[a_idx], probs[b_idx]);
            float prob = probs[idx];
            float3 co = make_float3(0.f, -ab.z, ab.y);
            if (length(co) < 1e-7) co = make_float3(-ab.y, ab.x, 0.f);
            float3 x = co / (length(co) + 1e-7) * odist;
            float3 y = cross(x, ab);
            int adaptive_n_rots = min(int(odist / res * (2 * M_PI)), n_rots);
            // int adaptive_n_rots = n_rots;
            for (int i = 0; i < adaptive_n_rots; i++) {
                float angle = i * 2 * M_PI / adaptive_n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 center_grid = (c + offset - make_float3(corner[0], corner[1], corner[2])) / res;
                if (center_grid.x < 0.01 || center_grid.y < 0.01 || center_grid.z < 0.01 ||
                    center_grid.x >= grid_x - 1.01 || center_grid.y >= grid_y - 1.01 || center_grid.z >= grid_z - 1.01) {
                    // atomicAdd(&grid_obj[20 * grid_y * grid_z + 34 * grid_z + 33], 0.5 * prob);
                    continue;
                }
                int3 center_grid_floor = make_int3(center_grid);
                int3 center_grid_ceil = center_grid_floor + 1;
                float3 residual = fracf(center_grid);

                float3 w0 = 1.f - residual;
                float3 w1 = residual;

                float lll = w0.x * w0.y * w0.z;
                float llh = w0.x * w0.y * w1.z;
                float lhl = w0.x * w1.y * w0.z;
                float lhh = w0.x * w1.y * w1.z;
                float hll = w1.x * w0.y * w0.z;
                float hlh = w1.x * w0.y * w1.z;
                float hhl = w1.x * w1.y * w0.z;
                float hhh = w1.x * w1.y * w1.z;

                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], lll * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], llh * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], lhl * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], lhh * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], hll * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], hlh * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], hhl * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], hhh * prob);
            }
        }
    }

    __global__ void vote_rotation(const float *points, const float *preds_rot, float *outputs_up,
            const int *point_idxs, int n_ppfs, int n_rots)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs && (preds_rot[idx] < 1.3962634 || preds_rot[idx] > 1.745329)) {
            float rot = preds_rot[idx];
            int a_idx = point_idxs[idx * 2];
            int b_idx = point_idxs[idx * 2 + 1];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 ab = a - b;
            ab /= (length(ab) + 1e-7);

            float3 co = make_float3(0.f, -ab.z, ab.y);
            if (length(co) < 1e-7) co = make_float3(-ab.y, ab.x, 0.f);
            float3 x = co / (length(co) + 1e-7);
            float3 y = cross(x, ab);

            for (int i = 0; i < n_rots; i++) {
                float angle = i * 2 * M_PI / n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 up = tan(rot) * offset + (tan(rot) > 0 ? ab : -ab);
                // float3 up = cos(rot) * ab + sin(rot) * offset;
                up = up / (length(up) + 1e-7);

                outputs_up[(idx * n_rots + i) * 3 + 0] = up.x;
                outputs_up[(idx * n_rots + i) * 3 + 1] = up.y;
                outputs_up[(idx * n_rots + i) * 3 + 2] = up.z;
            }
        }
    }

    __global__ void backvote(const float *points, const float *outputs, bool *output_mask, const int *point_idxs,
            const float *corner, const float res, int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z,
            const float *gt_center, const float tol)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            // output_mask[idx] = true;
            float proj_len = outputs[idx * 2];
            float odist = outputs[idx * 2 + 1];
            int a_idx = point_idxs[idx * 2];
            int b_idx = point_idxs[idx * 2 + 1];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 ab = a - b;
            ab /= (length(ab) + 1e-7);
            float3 c = a - ab * proj_len;

            float3 co = make_float3(0.f, -ab.z, ab.y);
            float3 x = co / (length(co) + 1e-7) * odist;
            float3 y = cross(x, ab);

            // out_offsets[idx] = make_float3(0, 0, 0);
            output_mask[idx] = false;
            int adaptive_n_rots = min(int(odist / res * (2 * M_PI)), n_rots);
            for (int i = 0; i < adaptive_n_rots; i++) {
                float angle = i * 2 * M_PI / adaptive_n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 pred_center = c + offset;
                if (length(pred_center - make_float3(gt_center[0], gt_center[1], gt_center[2])) > tol) continue;
                float3 center_grid = (pred_center - make_float3(corner[0], corner[1], corner[2])) / res;
                if (center_grid.x < 0 || center_grid.y < 0 || center_grid.z < 0 ||
                    center_grid.x >= grid_x - 1 || center_grid.y >= grid_y - 1 || center_grid.z >= grid_z - 1) {
                    continue;
                }
                // out_offsets[idx] = -offset;
                output_mask[idx] = true;
                break;
            }
        }
    }

} // namespace cc2d

torch::Tensor vote_translation(
        const torch::Tensor &points, const torch::Tensor &outputs,
        const torch::Tensor &point_idxs, const torch::Tensor &corner, const float res, const int n_rots,
		const int grid_x, const int grid_y, const int grid_z) {
    AT_ASSERTM(points.is_cuda(), "points must be a CUDA tensor");
    AT_ASSERTM(points.ndimension() == 2, "input must be a  [N, 3] shape");
    AT_ASSERTM(points.scalar_type() == torch::kFloat32, "input must be a float32 type");
    AT_ASSERTM(point_idxs.scalar_type() == torch::kInt32, "point index must be a int32 type");

    const int32_t sample_num = point_idxs.size(-2);
    auto float32_options = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());
    torch::Tensor grid_obj = torch::zeros({grid_x, grid_y, grid_z}, float32_options);
    torch::Tensor probs = torch::ones({sample_num}, float32_options);

    const dim3 grid((sample_num + THREAD_NUM - 1) / THREAD_NUM);
    const dim3 block(THREAD_NUM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    voting::vote_translation<<<grid, block, 0, stream>>>(
        points.data_ptr<float>(), outputs.data_ptr<float>(), probs.data_ptr<float>(),
        point_idxs.data_ptr<int32_t>(), grid_obj.data_ptr<float>(), corner.data_ptr<float>(),
        res, sample_num, n_rots, grid_x, grid_y, grid_z
    );
    return grid_obj;
}


torch::Tensor vote_rotation(
        const torch::Tensor &points, const torch::Tensor &preds_rot, const torch::Tensor &point_idxs,
        const int n_rots) {
    AT_ASSERTM(points.is_cuda(), "points must be a CUDA tensor");
    AT_ASSERTM(points.ndimension() == 2, "input must be a  [N, 3] shape");
    AT_ASSERTM(points.scalar_type() == torch::kFloat32, "input must be a float32 type");
    AT_ASSERTM(point_idxs.scalar_type() == torch::kInt32, "point index must be a int32 type");

    const int32_t sample_num = point_idxs.size(-2);
    auto outputs_options = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());
    torch::Tensor outputs_up = torch::zeros({sample_num, n_rots, 3}, outputs_options);

    const dim3 grid((sample_num + THREAD_NUM - 1) / THREAD_NUM);
    const dim3 block(THREAD_NUM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    voting::vote_rotation<<<grid, block, 0, stream>>>(
        points.data_ptr<float>(), preds_rot.data_ptr<float>(), outputs_up.data_ptr<float>(),
        point_idxs.data_ptr<int32_t>(), sample_num, n_rots);
    return outputs_up;
}


torch::Tensor backvote(
        const torch::Tensor &points, const torch::Tensor &outputs, const torch::Tensor &pred_center,
        const torch::Tensor &point_idxs, const torch::Tensor &corner, const float res,
        const int n_rots, const int grid_x, const int grid_y, const int grid_z, const float tol) {
    AT_ASSERTM(points.is_cuda(), "points must be a CUDA tensor");
    AT_ASSERTM(points.ndimension() == 2, "input must be a  [N, 3] shape");
    AT_ASSERTM(points.scalar_type() == torch::kFloat32, "input must be a float32 type");
    AT_ASSERTM(point_idxs.scalar_type() == torch::kInt32, "point index must be a int32 type");

    const int32_t sample_num = point_idxs.size(-2);
    auto outputs_options = torch::TensorOptions().dtype(torch::kBool).device(points.device());
    torch::Tensor output_mask = torch::zeros({sample_num}, outputs_options);

    const dim3 grid((sample_num + THREAD_NUM - 1) / THREAD_NUM);
    const dim3 block(THREAD_NUM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    voting::backvote<<<grid, block, 0, stream>>>(
        points.data_ptr<float>(), outputs.data_ptr<float>(), output_mask.data_ptr<bool>(),
        point_idxs.data_ptr<int32_t>(), corner.data_ptr<float>(), res, sample_num, n_rots,
        grid_x, grid_y, grid_z, pred_center.data_ptr<float>(), tol * res);
    return output_mask;
}

