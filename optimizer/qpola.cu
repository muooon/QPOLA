// =========================================================================
// QPOLA カーネルコア v1.0.2 260720 by muooon https://github.com/muooon/QPOLA
// =========================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// 各種量子化･低ビット型用のヘッダー
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// =========================================================================
// 型ごとの固有処理を吸収するヘルパー(型トレイト)
// =========================================================================

// 1. メモリからの読込み時：各量子化型から高精度FP32へ変換
__device__ __forceinline__ float to_float(float val) { return val; }
__device__ __forceinline__ float to_float(__half val) { return __half2float(val); }
__device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }
__device__ __forceinline__ float to_float(signed char val) { return static_cast<float>(val); }
__device__ __forceinline__ float to_float(__nv_fp8_e4m3 val) { return static_cast<float>(val); } // 演算子による自動変換
__device__ __forceinline__ float to_float(__nv_fp8_e5m2 val) { return static_cast<float>(val); } // 演算子による自動変換

// 2. メモリへの書き戻し時：FP32から元のターゲット型へキャスト(安全装置付き)
template <typename T> __device__ __forceinline__ T from_float(float val);
template <> __device__ __forceinline__ float from_float<float>(float val) { 
    return val; 
}
template <> __device__ __forceinline__ __half from_float<__half>(float val) { 
    // FP16の限界値 [-65504.0, 65504.0] でガード
    return __float2half(fmaxf(-65504.0f, fminf(65504.0f, val))); 
}
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) { 
    // BF16はFP32と同じ範囲まで表現できますが、異常値防止としてFP16と同じ範囲か、
    // あるいは少し広めの妥当な値(例: 65504.0f)で念のため縛っておくと安全です
    return __float2bfloat16(fmaxf(-65504.0f, fminf(65504.0f, val))); 
}
template <> __device__ __forceinline__ signed char from_float<signed char>(float val) { 
    // INT8有効範囲 [-128, 127] にクリップしオーバーフローを防ぐ (既存のまま)
    return static_cast<signed char>(fmaxf(-128.0f, fminf(127.0f, val))); 
}
template <> __device__ __forceinline__ __nv_fp8_e4m3 from_float<__nv_fp8_e4m3>(float val) { 
    // FP8 E4M3 の絶対的な最大値は 448.0f です(超えると一発アウトなので厳重にガード)
    return __nv_fp8_e4m3(fmaxf(-448.0f, fminf(448.0f, val))); 
}
template <> __device__ __forceinline__ __nv_fp8_e5m2 from_float<__nv_fp8_e5m2>(float val) { 
    // FP8 E5M2 の最大値は 57344.0f です
    return __nv_fp8_e5m2(fmaxf(-57344.0f, fminf(57344.0f, val))); 
}

// 3. 型ごとの適切な最小係数(下限)ヘルパ：7. 極座標型無次元更新へ
template <typename T> __device__ __forceinline__ float get_min_p_factor();

template <> __device__ __forceinline__ float get_min_p_factor<float>()           { return 0.01f; } // FP32は精密に
template <> __device__ __forceinline__ float get_min_p_factor<__half>()          { return 0.01f; } // FP16も精密に
template <> __device__ __forceinline__ float get_min_p_factor<__nv_bfloat16>()  { return 0.05f; } // BF16は少し粗い
template <> __device__ __forceinline__ float get_min_p_factor<signed char>()    { return 0.15f; } // INT8はかなり粗いので高く
template <> __device__ __forceinline__ float get_min_p_factor<__nv_fp8_e4m3>()  { return 0.10f; } // FP8も高めに
template <> __device__ __forceinline__ float get_min_p_factor<__nv_fp8_e5m2>()  { return 0.10f; }

// =========================================================================
// QPOLA コアロジック (__device__ 関数にしてカーネルからの呼出し許可)
// =========================================================================
template <typename T>
__device__ __forceinline__ void qpola_kernel_impl(
    T* p, 
    const T* g, 
    float base_lr, 
    float eps, 
    int n
) {
    // 共有メモリ：ブロック全体(8ワープ)の方向和
    // __shared__ float s_block_direction[BLOCK_SIZE / WARP_SIZE]; (元設計)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // 1. 境界内かどうかのフラグ
    bool is_active = (idx < n);

    // 有効なスレッドなら1.0f、無効なら0.0fをセット
    float active_count = is_active ? 1.0f : 0.0f; 

    // 各型(T)のポインタから正しいビット幅で読出し即座に内部計算用の float へ変換
    float g_val = is_active ? to_float(g[idx]) : 0.0f;
    float p_val = is_active ? to_float(p[idx]) : 0.0f;

    // 2. QJL代替(符号情報)
    float g_sign = (g_val > 0.0f) ? 1.0f : ((g_val < 0.0f) ? -1.0f : 0.0f);

    float warp_direction_sum = g_sign;
    float warp_p_abs_sum = fabsf(p_val);

    // ワープシャッフル(有効スレッド数も一緒に並列集計)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        warp_direction_sum += __shfl_down_sync(0xffffffff, warp_direction_sum, offset);
        warp_p_abs_sum     += __shfl_down_sync(0xffffffff, warp_p_abs_sum, offset);
        active_count       += __shfl_down_sync(0xffffffff, active_count, offset);
    }

    // ブロードキャスト
    float micro_direction_sum = __shfl_sync(0xffffffff, warp_direction_sum, 0);
    float warp_p_scale_sum    = __shfl_sync(0xffffffff, warp_p_abs_sum, 0);
    float warp_active_count   = __shfl_sync(0xffffffff, active_count, 0);

    // 【重要】 固定値(32)ではなく、実際の有効数で割る(ゼロ除算防止に max を挟む)
    float div_count = fmaxf(warp_active_count, 1.0f);
    float micro_direction_mean = micro_direction_sum / div_count; 
    float warp_p_scale         = fmaxf(warp_p_scale_sum / div_count, 1e-4f);

    // 3. マクロ集計(共有メモリへの書込み)
    // ※ブロック全体(8ワープ)の｢方向和｣と｢有効スレッド数和｣のための共有メモリ
    __shared__ float s_block_direction[BLOCK_SIZE / WARP_SIZE]; 
    __shared__ float s_block_active[BLOCK_SIZE / WARP_SIZE];    // ★追加

    if (lane == 0) {
        s_block_direction[warp_id] = micro_direction_sum;
        s_block_active[warp_id]    = warp_active_count;   // ★追加(各ワープの有効数を記録)
    }

    // ブロック内の全スレッドが揃うまで確実に待機
    __syncthreads(); 

    float block_direction_sum = 0.0f;
    float block_active_sum = 0.0f;                        // ★追加
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE / WARP_SIZE; ++i) {
        block_direction_sum += s_block_direction[i];
        block_active_sum    += s_block_active[i];         // ★追加
    }
    
    // 【重要】 BLOCK_SIZE(256)固定ではなく実際の有効総数で割る
    float macro_direction_mean = block_direction_sum / fmaxf(block_active_sum, 1.0f);

    // 4. すべての同期待ち･シャッフルの｢最後｣に書き込み対象外スレッドを保護
    if (!is_active) return;

    // 5. 極座標多重アライメント判定
    float micro_align = g_sign * micro_direction_mean; 
    float macro_align = g_sign * macro_direction_mean; 

    // 6. 極座標型アライメント制御(オーバーシュート防止型：滑らか･分岐なし：信頼度判定)
    // micro_align と macro_align が負(逆方向)に振れれば振れるほど conflict(不一致度)が大きくなる
    // 完全に一致(ともに最大値 1.0)のときは conflict = 0.0 になり、factor = 1.0f となる
    float diff_micro = 1.0f - micro_align;
    float diff_macro = 1.0f - macro_align;
    float conflict = (diff_micro + diff_macro) * 0.5f; // 0.0(完全一致) ～ 2.0(完全反転)

    // 不一致度に応じて滑らかに減衰(係数の 0.495f は conflict=2.0 のときに最低値 0.01 になる調整値)
    float adaptation_factor = 1.0f - conflict * 0.495f;

    // 最低値を 0.01(1%)、最高値を 1.0(100%)に安全クリップ(分岐なしの組み込み関数)
    adaptation_factor = fminf(fmaxf(adaptation_factor, 0.01f), 1.0f);

    // 7. 極座標型無次元更新(型に応じて係数を自動切り替え)
    float min_ratio = get_min_p_factor<T>(); 
    // float local_p_factor = fmaxf(log1pf((float)fabsf(p_val)), log1pf((float)warp_p_scale) * min_ratio);
    // データ型(T)が許容できる｢落差の最大限｣(1 / min_ratio)でブレーキ閾値を自動逆算
    // 【無次元化】 周囲のスケール(warp_p_scale)を基準にブレーキの強さを自律決定する
    // 係数に｢min_ratio｣をそのまま使うことで、型に応じた最適なセーフティネットが自動構成される
    float scale_inv = 1.0f / fmaxf(warp_p_scale, 1e-4f);
    float soft_p = fabsf(p_val) / (1.0f + min_ratio * fabsf(p_val) * scale_inv) ;
    float soft_warp = warp_p_scale / (1.0f + min_ratio * warp_p_scale * scale_inv);
    // アーキテクチャやハードウェアなどに依存しない log 的制動を統合する
    float local_p_factor = fmaxf(soft_p, soft_warp * min_ratio);
    float update_scale = local_p_factor * adaptation_factor;

    // 8. パラメータの更新と再量子化書戻し
    float next_p = p_val - (base_lr * g_sign * update_scale);
    p[idx] = from_float<T>(next_p);
}

// =========================================================================
// Python (ctypes) 側からデータ型に応じて明示的に呼び分ける外部エントリーポイント
// =========================================================================
extern "C" {

__global__ void qpola_kernel_fp32(float* p, const float* g, float base_lr, float eps, int n) {
    qpola_kernel_impl<float>(p, g, base_lr, eps, n);
}

__global__ void qpola_kernel_fp16(__half* p, const __half* g, float base_lr, float eps, int n) {
    qpola_kernel_impl<__half>(p, g, base_lr, eps, n);
}

__global__ void qpola_kernel_bf16(__nv_bfloat16* p, const __nv_bfloat16* g, float base_lr, float eps, int n) {
    qpola_kernel_impl<__nv_bfloat16>(p, g, base_lr, eps, n);
}

__global__ void qpola_kernel_int8(signed char* p, const signed char* g, float base_lr, float eps, int n) {
    qpola_kernel_impl<signed char>(p, g, base_lr, eps, n);
}

__global__ void qpola_kernel_fp8_e4m3(__nv_fp8_e4m3* p, const __nv_fp8_e4m3* g, float base_lr, float eps, int n) {
    qpola_kernel_impl<__nv_fp8_e4m3>(p, g, base_lr, eps, n);
}

__global__ void qpola_kernel_fp8_e5m2(__nv_fp8_e5m2* p, const __nv_fp8_e5m2* g, float base_lr, float eps, int n) {
    qpola_kernel_impl<__nv_fp8_e5m2>(p, g, base_lr, eps, n);
}

} // extern "C"