import os
import sys
import torch
from torch.optim import Optimizer

"""
QPOLA v1.2.0 261015 Pure PyTorch Edition (fp32 Only / Moment-Free)
QPOLARIS (Quantization n Polar-Aligned Resetting Instant SGD)
量子化に強い、履歴ゼロ、空間協調 (極座標･QJL) による 自己適応型SGD
推奨学習率： 1e-4(Pre･LoRA) / 1e-6(微調整) 程度でお試しください
事前学習では｢切断正規分布｣による初期化を検討してください
この仕組みは瞬時的な｢勾配の分解と再構成｣を行います(複次的に｢VRAM負荷削減｣になりました)
"""

# 対応環境等：
# デバイス (CPU, NVIDIA CUDA, AMD ROCm/HIP, Intel XPU, Apple Metal/MPS)
# AMP (fp16 / bf16 / fp32) メモリ効率化キャスト
# Weight / Block (w/b) 動的サイズ調整･パディング保護
# JIT (TorchScript) / Triton コンパイル自動判別

# usage ／ 使い方
# --optimizer_type=optimizer.qpola.QPOLA


# QPOLA コアロジック (2層空間リダクション / 空間協調)
@torch.jit.script
def _qpola_pure_step_kernel_impl(
    p_fp32: torch.Tensor, 
    g_fp32: torch.Tensor, 
    lr: float, 
    eps: float,
    warp_size: int  # クラス側からキャッシュされた値を受け取る
) -> torch.Tensor:
    n = p_fp32.numel()
    if n == 0:
        return p_fp32

    g_val = torch.nan_to_num(g_fp32.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    p_val = torch.nan_to_num(p_fp32.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)

    # 動的クエリを行わず、渡された実態の warp_size をそのまま window_size に反映
    block_size = 256
    window_size = min(warp_size, max(1, n))
    warps_per_block = block_size // window_size

    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        g_padded = torch.cat([g_val, g_val.new_zeros(pad_len)])
        valid_padded = torch.cat([
            g_val.new_ones(n),
            g_val.new_zeros(pad_len),
        ])
    else:
        g_padded = g_val
        valid_padded = g_val.new_ones(n)

    total_padded_len = g_padded.numel()

    # ワープ単位(32要素)に分割
    num_warps = total_padded_len // window_size
    g_grouped = g_padded.view(num_warps, window_size)
    valid_grouped = valid_padded.view(num_warps, window_size)

    g_sign_grouped = torch.sign(g_grouped)
    valid_count = valid_grouped.sum(dim=-1, keepdim=True).clamp_min(1.0)
    micro_direction_mean = ((g_sign_grouped * valid_grouped).sum(dim=-1, keepdim=True) / valid_count)
    warp_g_scale = ((g_grouped.abs() * valid_grouped).sum(dim=-1, keepdim=True) / valid_count)

    # ブロック単位(256要素＝8ワープ分)でマクロ集計を再現
    num_blocks = num_warps // warps_per_block
    g_sign_blocks = g_sign_grouped.view(num_blocks, warps_per_block, window_size)
    valid_blocks = valid_grouped.view(num_blocks, warps_per_block, window_size)

    # ブロック内の方向総和 と 有効要素数の算出 (s_block_direction / s_block_active)
    # ブロック内はパディングも含めて常に固定サイズで処理
    block_direction_sum = (g_sign_blocks * valid_blocks).sum(dim=(1, 2), keepdim=True)
    block_active_count = valid_blocks.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    
    # ブロックごとのマクロ平均(各ブロック内の256要素で割る)
    macro_direction_mean_per_block = (block_direction_sum / block_active_count)
    
    # 元のワープ単位の形状にブロードキャストして戻す
    macro_direction_mean = macro_direction_mean_per_block.expand(num_blocks, warps_per_block, 1).reshape(num_warps, 1)

    # ミクロ・マクロのアライメント計算
    micro_align = g_sign_grouped * micro_direction_mean
    macro_align = g_sign_grouped * macro_direction_mean

    diff_micro = torch.clamp(1.0 - micro_align, min=0.0)
    diff_macro = torch.clamp(1.0 - macro_align, min=0.0)
    conflict = (diff_micro + diff_macro) * 0.5

    # フラット化して元の有効要素数 n に切り出し
    conflict_flat = conflict.view(-1)[:n]
    warp_g_scale_flat = warp_g_scale.expand_as(g_grouped).contiguous().view(-1)[:n]

    min_factor = 1e-5
    decay_rate = (1.0 - min_factor) * 0.5
    raw_adaptation = 1.0 - conflict_flat * decay_rate
    adaptation_factor = torch.clamp(raw_adaptation, min=min_factor, max=1.0)

    lim = 16.0
    g_hat_raw = g_val / (warp_g_scale_flat + eps)
    g_hat = torch.tanh(g_hat_raw / lim) * lim

    clamp_max = 3.4e38
    next_p = p_val - (lr * g_hat * adaptation_factor)
    next_p = torch.clamp(next_p, min=-clamp_max, max=clamp_max)

    next_p = torch.where(torch.isnan(next_p) | torch.isinf(next_p), p_val, next_p)
    return next_p.view_as(p_fp32)


# コンパイル処理 (Windows では安全に JIT Script、 Linux + Triton では compile)
_qpola_pure_step_kernel = None

# サポートされているアクセラレータ (CUDA, HIP, XPU) 利用可能チェック
has_accelerator = (
    torch.cuda.is_available() or 
    (hasattr(torch, 'xpu') and torch.xpu.is_available())
)

# アクセラレータがある環境でのみコンパイル / JIT化を試行
if has_accelerator:
    # Windows では Triton エラーを避けるため最初から Script 化
    if sys.platform.startswith('win'):
        try:
            _qpola_pure_step_kernel = torch.jit.script(_qpola_pure_step_kernel_impl)
        except Exception:
            _qpola_pure_step_kernel = _qpola_pure_step_kernel_impl
    else:
        if hasattr(torch, 'compile'):
            try:
                _qpola_pure_step_kernel = torch.compile(_qpola_pure_step_kernel_impl, mode="reduce-overhead")
            except Exception:
                _qpola_pure_step_kernel = None

        if _qpola_pure_step_kernel is None:
            try:
                _qpola_pure_step_kernel = torch.jit.script(_qpola_pure_step_kernel_impl)
            except Exception:
                _qpola_pure_step_kernel = _qpola_pure_step_kernel_impl

# アクセラレータがない場合 CPU、 MPS(mac) 等 や、コンパイルに失敗した場合にノーマル関数を使用
if _qpola_pure_step_kernel is None:
    _qpola_pure_step_kernel = _qpola_pure_step_kernel_impl


# PyTorch Optimizer クラス定義
class QPOLA(Optimizer):
    def __init__(self, params, 
                 lr=1e-3, 
                 eps=1e-8, 
                 low_vram=False,  
                 betas=None, 
                 weight_decay=None):
        defaults = dict(lr=lr, eps=eps)
        super(QPOLA, self).__init__(params, defaults)
        # betas, weight_decay 等は学習側の記述等を受け流すためのダミーです
        self.low_vram = low_vram
        # warp / wave size cache
        self._cached_warp_size = None

    def _get_warp_size(self, device):
        if self._cached_warp_size is not None:
            return self._cached_warp_size

        base_warp = 32
        dev_type = device.type
        
        if dev_type == 'cuda':
            try:
                props = torch.cuda.get_device_properties(device)
                if hasattr(props, 'warp_size'):
                    base_warp = props.warp_size
            except Exception:
                pass
        elif dev_type == 'xpu':
            try:
                props = torch.xpu.get_device_properties(device)
                # サブグループサイズなどのプロパティがあれば取得、なければ16や32など
                if hasattr(props, 'sub_group_sizes') and props.sub_group_sizes:
                    base_warp = props.sub_group_sizes[-1]
            except Exception:
                pass

        self._cached_warp_size = base_warp
        return base_warp

    @torch.no_grad()
    def step(self, closure=None):
        loss = torch.enable_grad()(closure)() if closure is not None else None

        for group in self.param_groups:
            lr, eps = group['lr'], group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                orig_dtype = p.dtype

                # 高速化：fp32 の場合は不要なキャストやコピーを回避
                p_fp32 = p if orig_dtype == torch.float32 else p.to(dtype=torch.float32)
                g_fp32 = g if orig_dtype == torch.float32 else g.to(dtype=torch.float32)

                # キャッシュ済みの warp_size を取得してカーネルに渡す
                warp_size = self._get_warp_size(p.device)

                # 計算実行 (Compiled / JIT Script)
                next_p = _qpola_pure_step_kernel(p_fp32, g_fp32, float(lr), float(eps), int(warp_size))

                # 書き戻し （fp16 / bf16 への自動キャスト書き戻し含む）
                if orig_dtype == torch.float32:
                    p.copy_(next_p)
                else:
                    p.copy_(next_p.to(dtype=orig_dtype))

        # VRAMキャッシュのクリーンアップ制御 (NVIDIA / AMD および Intel 対応)
        # 【選択式】 プールされた未使用VRAMキャッシュの完全解放(クリーンアップ) / 中級者以上向け
        # VRAMに余裕のない方は low_vram=True と設定してください (通常：False / 高速化)
        if self.low_vram:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                torch.xpu.empty_cache()
            elif hasattr(torch.backends, 'mps') and hasattr(torch.mps, 'empty_cache'):
                # Apple Silicon (Mac) 向け
                torch.mps.empty_cache()

        return loss


"""
 https://github.com/muooon/qpola
 True Gradient will guide you through it all; believing in it and continuing to move forward is what fosters growth.
 Don’t let the past control you—the noise within the past is the very source of your worries and suffering.
"""