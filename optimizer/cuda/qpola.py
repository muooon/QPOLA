import os
import ctypes
import sys
import torch
from torch.optim import Optimizer

"""
QPOLA v1.2.0 261015 CUDA Origin Edition (fp32 Only / Moment-Free)
QPOLARIS (Quantization n Polar-Aligned Resetting Instant SGD)
量子化に強い、履歴ゼロ、空間協調 (極座標･QJL) による 自己適応型SGD
推奨学習率： 1e-4(Pre･LoRA) / 1e-6(微調整) 程度でお試しください
事前学習では｢切断正規分布｣による初期化を検討してください
この仕組みは瞬時的な｢勾配の分解と再構成｣を行います(複次的に｢VRAM負荷削減｣になりました)
"""

# usage ／ 使い方
# --optimizer_type=optimizer.qpola.QPOLA
# Please place qpola.py and qpola_kernel.ptx in the same folder.

# compile ／ コンパイル
# Linux：
# nvcc -Xcompiler "-fPIC" -ptx -o qpola_kernel.ptx qpola.cu
# Windows：
# nvcc -Xcompiler "/utf-8" -ptx -o qpola_kernel.ptx qpola.cu


current_dir = os.path.dirname(os.path.abspath(__file__))
ptx_path = os.path.join(current_dir, "qpola_kernel.ptx")

if not os.path.exists(ptx_path):
    raise FileNotFoundError(f"QPOLA PTXファイルが見つかりません: {ptx_path}")

try:
    if sys.platform.startswith('win'):
        cuda_driver = ctypes.CDLL("nvcuda.dll")
    else:
        cuda_driver = ctypes.CDLL("libcuda.so")
except OSError:
    raise RuntimeError("CUDA Driver (nvcuda.dll / libcuda.so) が見つかりません")

# Driver API の関数の引数型を明示的に定義(クラッシュ防止)
cuda_driver.cuModuleLoadData.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), 
    ctypes.c_void_p]
cuda_driver.cuModuleGetFunction.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), 
    ctypes.c_void_p, ctypes.c_char_p,]
cuda_driver.cuLaunchKernel.argtypes = [
    ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,ctypes.c_uint, 
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

with open(ptx_path, "rb") as f:
    ptx_bytes = f.read()

CUDA_MODULES = {}
CUDA_KERNELS = {}

def get_cuda_kernel(device_index: int, kernel_name: bytes):
    global CUDA_MODULES, CUDA_KERNELS
    cache_key = (device_index, kernel_name)
    if cache_key in CUDA_KERNELS:
        return CUDA_KERNELS[cache_key]

    torch.cuda.set_device(device_index)

    if device_index not in CUDA_MODULES:
        module = ctypes.c_void_p()
        res = cuda_driver.cuModuleLoadData(ctypes.byref(module), ptx_bytes)
        if res != 0:
            raise RuntimeError(f"GPU:{device_index} でのモジュールロード失敗 (コード: {res})")
        CUDA_MODULES[device_index] = module
    else:
        module = CUDA_MODULES[device_index]

    kernel = ctypes.c_void_p()
    res = cuda_driver.cuModuleGetFunction(ctypes.byref(kernel), module, kernel_name)
    if res != 0:
        raise RuntimeError(f"GPU:{device_index} でのカーネル {kernel_name.decode()} 取得失敗 (コード: {res})")

    CUDA_KERNELS[cache_key] = kernel
    return kernel


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

    @torch.no_grad()
    def step(self, closure=None):
        loss = torch.enable_grad()(closure)() if closure is not None else None

        for group in self.param_groups:
            lr, eps = group['lr'], group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                orig_device = p.device
                orig_dtype = p.dtype
                is_cpu_tensor = not p.is_cuda

                # fp32専用化：自動的にfp32にキャストして処理
                p_fp32 = p if orig_dtype == torch.float32 else p.to(dtype=torch.float32)
                g_fp32 = g if orig_dtype == torch.float32 else g.to(dtype=torch.float32)

                # デバイスの調停
                if is_cpu_tensor:
                    target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
                    p_cuda = p_fp32.to(target_device)
                    g_cuda = g_fp32.to(target_device)
                else:
                    target_device = orig_device
                    p_cuda = p_fp32 if orig_dtype == torch.float32 else p_fp32.to(target_device)
                    g_cuda = g_fp32 if orig_dtype == torch.float32 else g_fp32.to(target_device)

                # メモリ連続性の保証と｢非連続テンソル｣への対策
                if not p_cuda.is_contiguous():
                    p_cuda = p_cuda.contiguous()
                if not g_cuda.is_contiguous():
                    g_cuda = g_cuda.contiguous()

                n = p_cuda.numel()
                device_idx = target_device.index if target_device.index is not None else 0
                
                # 常に fp32 用のカーネルを呼び出す
                k_name = b"qpola_kernel_fp32"
                kernel = get_cuda_kernel(device_idx, k_name)

                # 引数の構築
                p_ptr = ctypes.c_void_p(p_cuda.data_ptr())
                g_ptr = ctypes.c_void_p(g_cuda.data_ptr())
                c_lr = ctypes.c_float(lr)
                c_eps = ctypes.c_float(eps)
                c_n = ctypes.c_int(n)

                args = [
                    ctypes.byref(p_ptr),
                    ctypes.byref(g_ptr),
                    ctypes.byref(c_lr),
                    ctypes.byref(c_eps),
                    ctypes.byref(c_n)
                ]
                arg_arr = (ctypes.c_void_p * len(args))(*[ctypes.cast(a, ctypes.c_void_p) for a in args])

                stream = torch.cuda.current_stream(target_device).cuda_stream

                res = cuda_driver.cuLaunchKernel(
                    kernel,
                    (n + 255) // 256, 1, 1,  # gridDim
                    256, 1, 1,               # blockDim
                    0,                       # sharedMem
                    ctypes.c_void_p(stream), # stream
                    arg_arr,                 # kernelParams
                    None                     # extra
                )

                if res != 0:
                    raise RuntimeError(f"GPU:{device_idx} 内で {k_name.decode()} の実行に失敗 (コード: {res})")

                # 結果を元のテンソル(元の型・デバイス)に書き戻し
                if orig_dtype == torch.float32:
                    if is_cpu_tensor:
                        torch.cuda.current_stream(target_device).synchronize()
                        p.copy_(p_cuda)
                    else:
                        p.copy_(p_cuda)
                else:
                    # bf/fp16などからfp32で計算して戻す場合
                    p.copy_(p_cuda.to(dtype=orig_dtype))

                # 一時変数の解放
                if is_cpu_tensor or orig_dtype != torch.float32:
                    del p_cuda
                if not g.is_contiguous():
                    del g_cuda

        # VRAMキャッシュのクリーンアップ制御
        # 【選択式】 プールされた未使用VRAMキャッシュの完全解放(クリーンアップ) / 中級者以上向け
        # VRAMに余裕のない方は low_vram=True と設定してください (通常：False / 高速化)
        if self.low_vram and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss
        

"""
 https://github.com/muooon/qpola
 True Gradient will guide you through it all; believing in it and continuing to move forward is what fosters growth.
 Don’t let the past control you—the noise within the past is the very source of your worries and suffering.
"""