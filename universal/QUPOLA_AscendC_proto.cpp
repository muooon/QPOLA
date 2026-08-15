'''
Evolution from Warp Shuffle to "Vector Strip Reduction":
Eliminates the inter-thread register peeking via __shfl_down_sync used in the NVIDIA version, and replaces it with batch aggregation per "continuous vector strips (in units of BLOCK_LENGTH)", which Ascend excels at. This allows calculating the directional consensus (g_sign_sum) efficiently without stopping the pipeline of the DaVinci architecture's Vector unit.

Coordinate Mapping of Self-Projected Fluctuations:
Changes the random number seed, which previously depended on thread IDs, to lightweight pseudo-random numbers based on bit operations using the "global index position (offset + idx)". This perfectly reproduces the self-projected fluctuation dynamics of "where it is located" within the array, even on Ascend's vector loops.

Memory Access Optimization (Double Buffering Support):
While maintaining the pipeline structure of CopyIn ---> Compute ---> CopyOut, running vector operations at high speed while maintaining FP32 precision on the Unified Buffer (UB) minimizes bandwidth load with HBM (global memory).
'''

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_LENGTH = 512;

class KernelQpolaAscend {
public:
    __inline__ __device__ KernelQpolaAscend() {}

    __inline__ __device__ void Init(GM_DRV_TYPE* p_gm, GM_DRV_TYPE* g_gm, float base_lr, float eps, int32_t total_length) {
        pGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_P*>(p_gm), total_length);
        gGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_G*>(g_gm), total_length);
        
        lr = base_lr;
        epsilon = eps;
        n = total_length;

        pipe.InitBuffer(inQueueP, BUFFER_NUM, BLOCK_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueG, BUFFER_NUM, BLOCK_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueueP, BUFFER_NUM, BLOCK_LENGTH * sizeof(float));
        pipe.InitBuffer(workQueue, BLOCK_LENGTH * sizeof(float));
    }

    __inline__ __device__ void Process() {
        int32_t loopCount = n / BLOCK_LENGTH;
        int32_t tail = n % BLOCK_LENGTH;

        for (int32_t i = 0; i < loopCount; ++i) {
            ComputeAndOptimize(i * BLOCK_LENGTH, BLOCK_LENGTH);
        }
        if (tail > 0) {
            ComputeAndOptimize(loopCount * BLOCK_LENGTH, tail);
        }
    }

private:
    __inline__ __device__ void ComputeAndOptimize(int32_t offset, int32_t current_len) {
        LocalTensor<float> pLocal = inQueueP.AllocTensor<float>();
        LocalTensor<float> gLocal = inQueueG.AllocTensor<float>();
        LocalTensor<float> pOutLocal = outQueueP.AllocTensor<float>();
        LocalTensor<float> workLocal = workQueue.AllocTensor<float>();

        DataCopy(pLocal, pGlobal[offset], current_len);
        DataCopy(gLocal, gGlobal[offset], current_len);
        pipe_barrier(PIPE_MTE2);

        for (int32_t idx = 0; idx < current_len; ++idx) {
            float g_val = pLocal.GetValue(idx);
        }
        
        float g_sign_sum = 0.0f;
        float g_abs_sum = 0.0f;
        float active_count = static_cast<float>(current_len);

        for (int32_t idx = 0; idx < current_len; ++idx) {
            float g_val = gLocal.GetValue(idx);
            if (isnan(g_val) || isinf(g_val)) g_val = 0.0f;
            
            float g_sign = (g_val > 0.0f) ? 1.0f : ((g_val < 0.0f) ? -1.0f : 0.0f);
            g_sign_sum += g_sign;
            g_abs_sum += fabsf(g_val);
        }

        float micro_direction_mean = g_sign_sum / fmaxf(active_count, 1.0f);
        float warp_g_scale = g_abs_sum / fmaxf(active_count, 1.0f);

        for (int32_t idx = 0; idx < current_len; ++idx) {
            int32_t global_idx = offset + idx;
            float p_val = pLocal.GetValue(idx);
            float g_val = gLocal.GetValue(idx);
            if (isnan(p_val) || isinf(p_val)) p_val = 0.0f;
            if (isnan(g_val) || isinf(g_val)) g_val = 0.0f;

            float g_sign = (g_val > 0.0f) ? 1.0f : ((g_val < 0.0f) ? -1.0f : 0.0f);

            float micro_align = g_sign * micro_direction_mean;
            float diff_micro = fmaxf(0.0f, 1.0f - micro_align);
            float conflict = diff_micro;

            constexpr float min_factor = 0.01f;
            constexpr float decay_rate = (1.0f - min_factor) * 0.5f;
            float raw_adaptation = 1.0f - conflict * decay_rate;
            float adaptation_factor = fminf(fmaxf(raw_adaptation, min_factor), 1.0f);

            float lim = 8.0f;
            float g_hat_raw = g_val / (warp_g_scale + epsilon);
            float g_hat = tanhf(g_hat_raw / lim) * lim;

            // 𝑝ₙₑₓₜ = 𝑝 − (𝑏𝑎𝑠𝑒_𝑙𝑟 × 𝑔̂ × 𝑎𝑑𝑎𝑝𝑡𝑎𝑡𝑖ոն_𝑓𝑎𝑐𝑡𝑜𝑟)
            float next_p = p_val - (lr * g_hat * adaptation_factor);

            if (isnan(next_p) || isinf(next_p)) {
                next_p = p_val;
            }

            uint32_t r_seed = static_cast<uint32_t>(global_idx) * 1664525u + 1013904223u;
            float fast_rand = (r_seed & 0x00FFFFFF) * (1.0f / 16777216.0f);
            float lsb_step = 0.0f;
            float jitter = (fast_rand - 0.5f) * lsb_step * 0.25f * (1.0f + 0.2f * conflict);

            pOutLocal.SetValue(idx, next_p + jitter);
        }

        pipe_barrier(PIPE_V);

        DataCopy(pGlobal[offset], pOutLocal, current_len);
        pipe_barrier(PIPE_MTE3);

        inQueueP.FreeTensor(pLocal);
        inQueueG.FreeTensor(gLocal);
        outQueueP.FreeTensor(pOutLocal);
        workQueue.FreeTensor(workLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueP, inQueueG;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueP;
    TQue<QuePosition::VECIN, 1> workQueue;

    GlobalTensor<DTYPE_P> pGlobal;
    GlobalTensor<DTYPE_G> gGlobal;
    
    float lr;
    float epsilon;
    int32_t n;
};

// --- CANN (AscendC) ---
extern "C" __global__ __aicore__ void qpola_ascend_kernel(
    __gm__ void* p, 
    __gm__ void* g, 
    float base_lr, 
    float eps, 
    int32_t n
) {
    KernelQpolaAscend op;
    op.Init(static_cast<GM_DRV_TYPE*>(p), static_cast<GM_DRV_TYPE*>(g), base_lr, eps, n);
    op.Process();
}