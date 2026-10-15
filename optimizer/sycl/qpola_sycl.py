import os
import ctypes
import sys
import torch
from torch.optim import Optimizer

"""
QPOLA v1.2.0 261015 SYCL Multi-Vendor Edition (fp32 Only / Moment-Free)
QPOLARIS (Quantization n Polar-Aligned Resetting Instant SGD)
量子化に強い、履歴ゼロ、空間協調 (極座標･QJL) による 自己適応型SGD
対応環境：NVIDIA(CUDA)、AMD(ROCm / HIP)、Intel(iPEX / XPU)、(SYCL 2020 準拠)
推奨学習率： 1e-4(Pre･LoRA) / 1e-6(微調整) 程度でお試しください
事前学習では｢切断正規分布｣による初期化を検討してください
(この仕組みは瞬時的な｢勾配の分解と再構成｣を行います(複次的に｢VRAM負荷削減｣になりました)
"""

# usage ／ 使い方
# --optimizer_type=optimizer.qpola_sycl.QPOLA
# Please place qpola_sycl.py and qpola_kernel_sycl.so
# (or .dll on Windows) in the same folder.

# compile ／ コンパイル
# Please compile in a SYCL environment. / SYCL コンパイル環境が必要です
# Linux：
# icpx -fsycl -fPIC -shared -O3 qpola_sycl.cpp -o qpola_kernel_sycl.so
# Windows：
# icx-cl /fsycl /LD /O3 qpola_sycl.cpp /Fe:qpola_kernel_sycl.dll


# プラットフォームに応じたSYCL共有ライブラリの拡張子を自動選択
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_name = "qpola_kernel_sycl.dll" if sys.platform.startswith('win') else "qpola_kernel_sycl.so"
lib_path = os.path.join(current_dir, lib_name)

if not os.path.exists(lib_path):
    raise FileNotFoundError(f"QPOLA SYCL 共有ライブラリが見つかりません: {lib_path}")

# C++側のエントリーポイント (C-APIラッパー) の引数型バインド
try:
    sycl_lib = ctypes.CDLL(lib_path)
    sycl_kernel_func = sycl_lib.qpola_kernel_fp32_sycl_capi
    sycl_kernel_func.argtypes = [
        ctypes.c_void_p,  # p
        ctypes.c_void_p,  # g
        ctypes.c_float,   # base_lr
        ctypes.c_float,   # eps
        ctypes.c_int      # n
    ]
    sycl_kernel_func.restype = None
except Exception as e:
    raise RuntimeError(f"SYCL ライブラリのバインドに失敗しました ({lib_path}): {e}")


class QPOLA(Optimizer):
    def __init__(self, params, 
                 lr=1e-3, 
                 eps=1e-8, 
                 low_vram=False, 
                 betas=None, 
                 weight_decay=None):
        defaults = dict(lr=lr, eps=eps)
        super(QPOLA, self).__init__(params, defaults)
        self.low_vram = low_vram
        # betas, weight_decay 等は学習側の記述等を受け流すためのダミーです

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

                # 精度とデバイスの整合（AMP対応 / デバイス不一致の防護）
                p_fp32 = p if orig_dtype == torch.float32 else p.float()
                g_fp32 = g if (orig_dtype == torch.float32 and g.device == p.device) else g.to(device=p.device, dtype=torch.float32)

                # メモリ連続性の保証
                p_fp32 = p_fp32.contiguous() if not p_fp32.is_contiguous() else p_fp32
                g_fp32 = g_fp32.contiguous() if not g_fp32.is_contiguous() else g_fp32

                # C-API 呼び出し（C++ 側で default_queue.wait() 同期済み）
                sycl_kernel_func(
                    ctypes.c_void_p(p_fp32.data_ptr()),
                    ctypes.c_void_p(g_fp32.data_ptr()),
                    ctypes.c_float(lr),
                    ctypes.c_float(eps),
                    ctypes.c_int(p_fp32.numel())
                )

                # パラメータへの書き戻し（fp16 / bf16 への自動キャスト書き戻し含む）
                if orig_dtype != torch.float32 or p_fp32 is not p:
                    p.copy_(p_fp32)

                # 一時テンソルの明示的解放
                #if p_was_not_contiguous or orig_dtype != torch.float32:
                #    del p_fp32
                #if g_was_not_contiguous or orig_dtype != torch.float32:
                #    del g_fp32

        # VRAMキャッシュのクリーンアップ制御 (NVIDIA / AMD および Intel 対応)
        # 【選択式】 プールされた未使用VRAMキャッシュの完全解放(クリーンアップ) / 中級者以上向け
        # VRAMに余裕のない方は low_vram=True と設定してください (通常：False / 高速化)
        if self.low_vram:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # NVIDIA および AMD (ROCm) 共通
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                torch.xpu.empty_cache()   # Intel XPU

        return loss

"""
 https://github.com/muooon/qpola
 True Gradient will guide you through it all; believing in it and continuing to move forward is what fosters growth.
 Don’t let the past control you—the noise within the past is the very source of your worries and suffering.
"""