// =========================================================================
// QPOLA カーネルコア v1.2.0 (fp32 Only) by muooon https://github.com/muooon/QPOLA
// =========================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ハードウェアに準ず
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// =========================================================================
// fp32 TypeTraits
// =========================================================================
struct TypeTraits {
    static __device__ __forceinline__ float clamp_max() { return 3.4e38f; }
    static __device__ __forceinline__ float lsb_step()  { return 0.0f; }
    static __device__ __forceinline__ float lim_g_hat() { return 16.0f; }
    static __device__ __forceinline__ float min_factor(){ return 1e-5f; }
};

// =========================================================================
// QPOLA コアロジック (__device__)
// =========================================================================
__device__ __forceinline__ void qpola_kernel_impl(
    float* p, 
    const float* g, 
    float base_lr, 
    float eps, 
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // 境界内かどうかの判定と初期読み込み
    bool is_active = (idx < n);
    float active_count = is_active ? 1.0f : 0.0f; 

    // プロトコル化(Coalesced Access / Contiguous)
    float g_val = is_active ? g[idx] : 0.0f;
    float p_val = is_active ? p[idx] : 0.0f;

    // NaN / Inf ガード
    if (isnan(g_val) || isinf(g_val)) g_val = 0.0f;
    if (isnan(p_val) || isinf(p_val)) p_val = 0.0f;

    // 勾配の方向(符号)と局所スケールの抽出
    float g_sign = (g_val > 0.0f) ? 1.0f : ((g_val < 0.0f) ? -1.0f : 0.0f);

    float warp_direction_sum = g_sign;
    float warp_g_abs_sum     = fabsf(g_val);

    // ワープシャッフルによる 32 スレッド並列集計(重み p は集計しない)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        active_count        += __shfl_down_sync(0xffffffff, active_count, offset);
        warp_direction_sum += __shfl_down_sync(0xffffffff, warp_direction_sum, offset);
        warp_g_abs_sum     += __shfl_down_sync(0xffffffff, warp_g_abs_sum, offset);
    }

    // ブロードキャスト
    float warp_active_count   = __shfl_sync(0xffffffff, active_count, 0);
    float micro_direction_sum = __shfl_sync(0xffffffff, warp_direction_sum, 0);
    float warp_g_scale_sum    = __shfl_sync(0xffffffff, warp_g_abs_sum, 0);

    float div_count = fmaxf(warp_active_count, 1.0f);
    float micro_direction_mean = micro_direction_sum / div_count; 
    float warp_g_scale         = warp_g_scale_sum / div_count; // 周囲の勾配平均スケール

    // マクロ集計(共有メモリによる 256 スレッドブロック集計)
    __shared__ float s_block_direction[BLOCK_SIZE / WARP_SIZE]; 
    __shared__ float s_block_active[BLOCK_SIZE / WARP_SIZE]; 

    if (lane == 0) {
        s_block_direction[warp_id] = micro_direction_sum;
        s_block_active[warp_id]    = warp_active_count;
    }

    __syncthreads(); 

    float block_direction_sum = 0.0f;
    float block_active_sum    = 0.0f; 
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE / WARP_SIZE; ++i) {
        block_direction_sum += s_block_direction[i];
        block_active_sum    += s_block_active[i];
    }

    float macro_direction_mean = block_direction_sum / fmaxf(block_active_sum, 1.0f);

    // 範囲外スレッドの保護(同期待ち完了後)
    if (!is_active) return;

    // 空間アライメント(一致度) / Conflict(衝突度) 算出
    float micro_align = g_sign * micro_direction_mean; 
    float macro_align = g_sign * macro_direction_mean; 

    float diff_micro = fmaxf(0.0f, 1.0f - micro_align);
    float diff_macro = fmaxf(0.0f, 1.0f - macro_align);
    float conflict   = (diff_micro + diff_macro) * 0.5f; // 0.0(完全一致) ～ 2.0(完全反転)

    // 減衰係数の算出
    const float min_factor = TypeTraits::min_factor();
    const float decay_rate = (1.0f - min_factor) * 0.5f;
    
    // Conflict に応じ 1.0 〜 min_factor の間でクランプ
    float raw_adaptation = 1.0f - conflict * decay_rate;
    float adaptation_factor = fminf(fmaxf(raw_adaptation, min_factor), 1.0f);

    // 勾配の無次元化 と 純粋な無次元更新
    float lim = TypeTraits::lim_g_hat();
    float g_hat_raw = g_val / (warp_g_scale + eps);
    float g_hat = tanhf(g_hat_raw / lim) * lim; // 無次元化勾配を上限で安全に飽和

    float max_v = TypeTraits::clamp_max();
    float next_p = p_val - (base_lr * g_hat * adaptation_factor);
    next_p = fmaxf(-max_v, fminf(max_v, next_p));

    // 書き戻し時の安全性保持
    if (isnan(next_p) || isinf(next_p)) next_p = p_val;
    p[idx] = next_p;
}

// =========================================================================
// Python (ctypes) 側からの外部エントリーポイント
// =========================================================================
extern "C" {

__global__ void qpola_kernel_fp32(float* p, const float* g, float base_lr, float eps, int n) {
    qpola_kernel_impl(p, g, base_lr, eps, n);
}

} // extern "C"