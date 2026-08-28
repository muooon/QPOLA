// =========================================================================
// QPOLA カーネルコア v1.0.5 (Intel XPU / SYCL 動的サブグループ対応版)
// =========================================================================

#include <sycl/sycl.hpp>
#include <cmath>

#define BLOCK_SIZE 256

// =========================================================================
// 型ごとの固有処理を吸収するヘルパー (型トレイト)
// =========================================================================

template <typename T>
inline float to_float(T val) {
    return static_cast<float>(val);
}

inline float to_float(float val)              { return val; }
inline float to_float(sycl::half val)         { return static_cast<float>(val); }
inline float to_float(sycl::ext::oneapi::bfloat16 val) { return static_cast<float>(val); }
inline float to_float(signed char val)        { return static_cast<float>(val); }

template <typename T> struct TypeTraits;

template <> struct TypeTraits<float> {
    static inline float clamp_max() { return 3.4e38f; }
    static inline float lsb_step()  { return 0.0f; }
    static inline float lim_g_hat() { return 16.0f; }
};

template <> struct TypeTraits<sycl::half> {
    static inline float clamp_max() { return 65504.0f; }
    static inline float lsb_step()  { return 0.0f; }
    static inline float lim_g_hat() { return 8.0f; }
};

template <> struct TypeTraits<sycl::ext::oneapi::bfloat16> {
    static inline float clamp_max() { return 3.4e38f; }
    static inline float lsb_step()  { return 0.0f; }
    static inline float lim_g_hat() { return 8.0f; }
};

template <> struct TypeTraits<signed char> {
    static inline float clamp_max() { return 127.0f; }
    static inline float lsb_step()  { return 1.0f; }
    static inline float lim_g_hat() { return 2.0f; }
};

inline float fast_rand_01(uint32_t global_id) {
    uint32_t x = global_id * 1664525u + 1013904223u;
    return (x & 0x00FFFFFF) * (1.0f / 16777216.0f);
}

template <typename T>
inline float apply_quant_jitter(float val, float conflict, uint32_t global_id) {
    float lsb = TypeTraits<T>::lsb_step();
    float jitter_scale = lsb * 0.25f * (1.0f + 0.2f * conflict);
    return val + (fast_rand_01(global_id) - 0.5f) * jitter_scale;
}

template <typename T> 
inline T from_float(float val, float conflict, uint32_t global_id);

template <> 
inline float from_float<float>(float val, float conflict, uint32_t global_id) {
    return val;
}

template <> 
inline sycl::half from_float<sycl::half>(float val, float conflict, uint32_t global_id) {
    float max_v = TypeTraits<sycl::half>::clamp_max();
    return static_cast<sycl::half>(sycl::fmax(-max_v, sycl::fmin(max_v, val)));
}

template <> 
inline sycl::ext::oneapi::bfloat16 from_float<sycl::ext::oneapi::bfloat16>(float val, float conflict, uint32_t global_id) {
    float max_v = TypeTraits<sycl::ext::oneapi::bfloat16>::clamp_max();
    return static_cast<sycl::ext::oneapi::bfloat16>(sycl::fmax(-max_v, sycl::fmin(max_v, val)));
}

template <> 
inline signed char from_float<signed char>(float val, float conflict, uint32_t global_id) {
    float max_v = TypeTraits<signed char>::clamp_max();
    float clamped = sycl::fmax(-max_v, sycl::fmin(max_v, val));
    return static_cast<signed char>(apply_quant_jitter<signed char>(clamped, conflict, global_id));
}

// =========================================================================
// QPOLA コアロジック (実行時サブグループサイズ対応)
// =========================================================================
template <typename T>
inline void qpola_kernel_impl(
    sycl::nd_item<1> item,
    T* p, 
    const T* g, 
    float base_lr, 
    float eps, 
    int n,
    float* s_block_direction, 
    float* s_block_active     
) {
    int idx = item.get_global_id(0);
    auto sg = item.get_sub_group();
    
    // 実行時にデバイスからサブグループサイズを取得
    uint32_t sub_group_size = sg.get_max_local_range()[0];
    int lane = sg.get_local_id()[0];
    int warp_id = item.get_local_id(0) / sub_group_size;
    int num_warps = item.get_local_range(0) / sub_group_size;

    bool is_active = (idx < n);
    float active_count = is_active ? 1.0f : 0.0f; 

    float g_val = is_active ? to_float(g[idx]) : 0.0f;
    float p_val = is_active ? to_float(p[idx]) : 0.0f;

    if (sycl::isnan(g_val) || sycl::isinf(g_val)) g_val = 0.0f;
    if (sycl::isnan(p_val) || sycl::isinf(p_val)) p_val = 0.0f;

    float g_sign = (g_val > 0.0f) ? 1.0f : ((g_val < 0.0f) ? -1.0f : 0.0f);

    float warp_direction_sum = g_sign;
    float warp_g_abs_sum     = sycl::fabs(g_val);

    // 動的サブグループサイズに対応したリダクション
    active_count         = sycl::reduce_over_group(sg, active_count, sycl::plus<>());
    warp_direction_sum   = sycl::reduce_over_group(sg, warp_direction_sum, sycl::plus<>());
    warp_g_abs_sum       = sycl::reduce_over_group(sg, warp_g_abs_sum, sycl::plus<>());

    float warp_active_count   = sg.broadcast(active_count, 0);
    float micro_direction_sum = sg.broadcast(warp_direction_sum, 0);
    float warp_g_scale_sum    = sg.broadcast(warp_g_abs_sum, 0);

    float div_count = sycl::fmax(warp_active_count, 1.0f);
    float micro_direction_mean = micro_direction_sum / div_count; 
    float warp_g_scale         = warp_g_scale_sum / div_count; 

    if (lane == 0) {
        s_block_direction[warp_id] = micro_direction_sum;
        s_block_active[warp_id]    = warp_active_count;
    }

    item.barrier(sycl::access::fence_space::local_space); 

    float block_direction_sum = 0.0f;
    float block_active_sum    = 0.0f; 
    for (int i = 0; i < num_warps; ++i) {
        block_direction_sum += s_block_direction[i];
        block_active_sum    += s_block_active[i];
    }

    float macro_direction_mean = block_direction_sum / sycl::fmax(block_active_sum, 1.0f);

    if (!is_active) return;

    float micro_align = g_sign * micro_direction_mean; 
    float macro_align = g_sign * macro_direction_mean; 

    float diff_micro = sycl::fmax(0.0f, 1.0f - micro_align);
    float diff_macro = sycl::fmax(0.0f, 1.0f - macro_align);
    float conflict   = (diff_micro + diff_macro) * 0.5f; 

    constexpr float min_factor = 0.01f;
    constexpr float decay_rate = (1.0f - min_factor) * 0.5f;
    
    float raw_adaptation = 1.0f - conflict * decay_rate;
    float adaptation_factor = sycl::fmin(sycl::fmax(raw_adaptation, min_factor), 1.0f);

    float lim = TypeTraits<T>::lim_g_hat();
    float g_hat_raw = g_val / (warp_g_scale + eps);
    float g_hat = sycl::tanh(g_hat_raw / lim) * lim; 

    float next_p = p_val - (base_lr * g_hat * adaptation_factor);

    if (sycl::isnan(next_p) || sycl::isinf(next_p)) next_p = p_val;
    p[idx] = from_float<T>(next_p, conflict, static_cast<uint32_t>(idx));
}

// =========================================================================
// 外部公開用エントリーポイント (実行時サブグループサイズの動的割当て)
// =========================================================================
extern "C" {

#define DEFINE_QPOLA_ENTRY(SUFFIX, TYPE) \
void qpola_kernel_##SUFFIX(sycl::queue& q, TYPE* p, const TYPE* g, float base_lr, float eps, int n) { \
    auto device = q.get_device(); \
    auto sub_group_sizes = device.get_info<sycl::info::device::sub_group_sizes>(); \
    size_t active_sg_size = sub_group_sizes.empty() ? 16 : sub_group_sizes[0]; \
    size_t num_warps_per_block = BLOCK_SIZE / active_sg_size; \
    \
    q.submit([=](sycl::handler& h) { \
        sycl::local_accessor<float, 1> s_dir(sycl::range<1>(num_warps_per_block), h); \
        sycl::local_accessor<float, 1> s_act(sycl::range<1>(num_warps_per_block), h); \
        h.parallel_for( \
            sycl::nd_range<1>(sycl::range<1>(((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)), \
            [=](sycl::nd_item<1> item) { \
                qpola_kernel_impl<TYPE>(item, p, g, base_lr, eps, n, s_dir.get_pointer(), s_act.get_pointer()); \
            } \
        ); \
    }); \
}

DEFINE_QPOLA_ENTRY(fp32,   float)
DEFINE_QPOLA_ENTRY(fp16,   sycl::half)
DEFINE_QPOLA_ENTRY(bf16,   sycl::ext::oneapi::bfloat16)
DEFINE_QPOLA_ENTRY(int8,   signed char)

#undef DEFINE_QPOLA_ENTRY

} // extern "C"