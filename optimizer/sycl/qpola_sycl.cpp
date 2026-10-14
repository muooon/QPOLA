// =========================================================================
// QPOLA カーネルコア v1.2.0 261010 (fp32 Only / Dynamic Sub-group SYCL 2020)
// by muooon https://github.com/muooon/QPOLA
// =========================================================================

#include <sycl/sycl.hpp>
#include <cmath>

#define BLOCK_SIZE 256 // ブロックあたりのスレッド数 (固定)

// =========================================================================
// fp32 TypeTraits
// =========================================================================
struct TypeTraits {
    static constexpr float clamp_max() { return 3.4e38f; }
    static constexpr float lim_g_hat() { return 16.0f; }
    static constexpr float min_factor(){ return 1e-5f; }
};

// =========================================================================
// QPOLA コアロジック (SYCL デバイス側処理 / 動的 warp/block 適応型)
// =========================================================================
inline void qpola_kernel_impl(
    float* p, 
    const float* g, 
    float base_lr, 
    float eps, 
    int n,
    sycl::nd_item<1> item,
    float* s_block_direction,
    float* s_block_active
) {
    int idx = item.get_global_id(0);
    auto sg = item.get_sub_group();
    
    // ハードウェアごとのサブグループサイズ [動的取得] (Warp / Wavefront size, etc)
    // NVIDIA: 32固定, AMD: 32 または 64, Intel: ハードウェア依存で可変 (例: 16, 32)
    int warp_size = sg.get_local_range()[0];
    int lane = sg.get_local_id()[0];
    int warp_id = item.get_local_id(0) / warp_size;

    // 境界内かどうかの判定と初期読込み
    bool is_active = (idx < n);
    float active_count = is_active ? 1.0f : 0.0f; 

    // プロトコル化 (Coalesced Access / Contiguous)
    float g_val = is_active ? g[idx] : 0.0f;
    float p_val = is_active ? p[idx] : 0.0f;

    // NaN / Inf ガード
    if (sycl::isnan(g_val) || sycl::isinf(g_val)) g_val = 0.0f;
    if (sycl::isnan(p_val) || sycl::isinf(p_val)) p_val = 0.0f;

    // 勾配の方向 (符号) と 局所スケールの抽出
    float g_sign = (g_val > 0.0f) ? 1.0f : ((g_val < 0.0f) ? -1.0f : 0.0f);

    // SYCL 2020 標準のリダクションを使用 (動的サブグループサイズに自動適合)
    float warp_active_count  = sycl::reduce_over_group(sg, active_count, sycl::plus<float>());
    float micro_direction_sum = sycl::reduce_over_group(sg, g_sign, sycl::plus<float>());
    float warp_g_scale_sum    = sycl::reduce_over_group(sg, sycl::fabs(g_val), sycl::plus<float>());

    float div_count = sycl::fmax(warp_active_count, 1.0f);
    float micro_direction_mean = micro_direction_sum / div_count; 
    float warp_g_scale         = warp_g_scale_sum / div_count; // 周囲の勾配平均スケール

    // マクロ集計 (各ワープの先頭スレッドが代表してローカルメモリへ書き込み)
    if (lane == 0) {
        s_block_direction[warp_id] = micro_direction_sum;
        s_block_active[warp_id]    = warp_active_count;
    }

    item.barrier(sycl::access::fence_space::local_space);

    // ブロック内の全ワープ数を動的に換算 (BLOCK_SIZE / warp_size)
    int num_warps = item.get_local_range(0) / warp_size;
    float block_direction_sum = 0.0f;
    float block_active_sum    = 0.0f; 

    for (int i = 0; i < num_warps; ++i) {
        block_direction_sum += s_block_direction[i];
        block_active_sum    += s_block_active[i];
    }

    float macro_direction_mean = block_direction_sum / sycl::fmax(block_active_sum, 1.0f);

    // 範囲外スレッドの保護
    if (!is_active) return;

    // 空間アライメント / Conflict 算出
    float micro_align = g_sign * micro_direction_mean; 
    float macro_align = g_sign * macro_direction_mean; 

    float diff_micro = sycl::fmax(0.0f, 1.0f - micro_align);
    float diff_macro = sycl::fmax(0.0f, 1.0f - macro_align);
    float conflict   = (diff_micro + diff_macro) * 0.5f;

    // 減衰係数の算出
    const float min_factor = TypeTraits::min_factor();
    const float decay_rate = (1.0f - min_factor) * 0.5f;
    
    float raw_adaptation = 1.0f - conflict * decay_rate;
    float adaptation_factor = sycl::fmin(sycl::fmax(raw_adaptation, min_factor), 1.0f);

    // 勾配の無次元化と更新
    float lim = TypeTraits::lim_g_hat();
    float g_hat_raw = g_val / (warp_g_scale + eps);
    float g_hat = sycl::tanh(g_hat_raw / lim) * lim;

    float max_v = TypeTraits::clamp_max();
    float next_p = p_val - (base_lr * g_hat * adaptation_factor);
    next_p = sycl::fmax(-max_v, sycl::fmin(max_v, next_p));

    if (sycl::isnan(next_p) || sycl::isinf(next_p)) next_p = p_val;
    p[idx] = next_p;
}

// =========================================================================
// ホスト側からのディスパッチ関数 (動的共有メモリ確保対応)
// =========================================================================
extern "C" {

void qpola_kernel_fp32_sycl(sycl::queue& q, float* p, const float* g, float base_lr, float eps, int n) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t global_range = num_blocks * BLOCK_SIZE;
    size_t local_range = BLOCK_SIZE;

    q.submit([&](sycl::handler& cgh) {
        // 最悪ケース (最小サブグループサイズ) を想定した最大ワープ数を安全に確保
        // 通常は BLOCK_SIZE / 最小サブグループサイズ (例：256 / 16 = 16 要素分)
        // どんな小さいサブグループ (最小1) が来てもオーバーフローしないよう BLOCK_SIZE を安全に確保
        sycl::local_accessor<float, 1> s_dir(sycl::range<1>(BLOCK_SIZE), cgh);
        sycl::local_accessor<float, 1> s_act(sycl::range<1>(BLOCK_SIZE), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> item) {
                qpola_kernel_impl(
                    p, g, base_lr, eps, n, item,
                    s_dir.get_multi_ptr<sycl::access::decorated::no>().get(),
                    s_act.get_multi_ptr<sycl::access::decorated::no>().get()
                );
            }
        );
    });
}

// Python ctypes 用 C-API エントリーポイント
void qpola_kernel_fp32_sycl_capi(float* p, const float* g, float base_lr, float eps, int n) {
    // SYCL キューを内部保持 (デフォルトデバイスを選択)
    static sycl::queue default_queue;

    // カーネル起動
    qpola_kernel_fp32_sycl(default_queue, p, g, base_lr, eps, n);

    // 非同期実行の完了待機 (PyTorch 側のコピー処理とのレースコンディションを防止)
    default_queue.wait();
}
} // extern "C"