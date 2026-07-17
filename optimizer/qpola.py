import os
import ctypes
import sys
import torch
from torch.optim import Optimizer

"""
QPOLA (v1.0.1 / Moment-Free) fp8/int8 対応済  ※ CUDA特性のため4bit未対応
QPOLARIS (Quantization n Polar-Aligned Resetting Instant SGD) 
量子化に強い、履歴ゼロ、空間協調(極座標･QJL)による自己適応型SGD

SDXL / Diffusion Fine-Tuning (FT)：1e-2(1e-1 〜 1e-3) LoRA：1e-1(〜1e-3)   
Transformer Full Fine-Tuning (FT)：1e-3(1e-2 〜 1e-4) LoRA：1e-2(〜1e-4) 

usage ／ 使い方
--optimizer_type=optimizer.qpola.QPOLA
Please place qpola.py and qpola_kernel.ptx in the same folder.
"""

# パスの解決と Driver API のロード
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

with open(ptx_path, "rb") as f:
    ptx_bytes = f.read()

# デバイス･型ごとのカーネル関数キャッシュマップ
CUDA_KERNELS = {}

def get_cuda_kernel(device_index: int, kernel_name: bytes):
    global CUDA_KERNELS
    cache_key = (device_index, kernel_name)
    if cache_key in CUDA_KERNELS:
        return CUDA_KERNELS[cache_key]

    torch.cuda.set_device(device_index)
    module = ctypes.c_void_p()
    res = cuda_driver.cuModuleLoadData(ctypes.byref(module), ptx_bytes)
    if res != 0:
        raise RuntimeError(f"GPU:{device_index} でのモジュールロード失敗 (コード: {res})")
    
    kernel = ctypes.c_void_p()
    res = cuda_driver.cuModuleGetFunction(ctypes.byref(kernel), module, kernel_name)
    if res != 0:
        raise RuntimeError(f"GPU:{device_index} でのカーネル {kernel_name.decode()} 取得失敗 (コード: {res})")

    CUDA_KERNELS[cache_key] = kernel
    return kernel


class QPOLA(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, eps=eps)
        super(QPOLA, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                orig_device = p.device
                orig_dtype = p.dtype
                is_cpu_tensor = not p.is_cuda

                # ーーー 1. PyTorch型からCUDAカーネル関数名への自動マッピング ーーー
                # ※ 注：特殊なFP8型(Float8e4m3fn等)は文字列判定等で柔軟にキャッチ
                dtype_str = str(orig_dtype)
                if orig_dtype == torch.float32:
                    k_name = b"qpola_kernel_fp32"
                elif orig_dtype == torch.float16:
                    k_name = b"qpola_kernel_fp16"
                elif orig_dtype == torch.bfloat16:
                    k_name = b"qpola_kernel_bf16"
                elif orig_dtype == torch.int8:
                    k_name = b"qpola_kernel_int8"
                elif "e4m3" in dtype_str:
                    k_name = b"qpola_kernel_fp8_e4m3"
                elif "e5m2" in dtype_str:
                    k_name = b"qpola_kernel_fp8_e5m2"
                else:
                    raise NotImplementedError(f"QPOLAは現在、型 {orig_dtype} をサポートしていません")

                # ーーー 2. デバイスまたぎの完全調停(カレントGPUへ集約) ーーー
                if is_cpu_tensor:
                    # CPU配置の場合は現在アクティブなカレントGPUデバイスへ強制転送
                    target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
                    p_cuda = p.to(target_device)
                    g_cuda = g.to(target_device)
                else:
                    target_device = orig_device
                    p_cuda = p
                    g_cuda = g

                # メモリ上の連続性を100%保証(転送･コピー時にアドレスを整列)
                if not p_cuda.is_contiguous():
                    p_cuda = p_cuda.contiguous()
                if not g_cuda.is_contiguous():
                    g_cuda = g_cuda.contiguous()

                n = p_cuda.numel()
                device_idx = target_device.index if target_device.index is not None else 0

                # 該当デバイス･該当型に対応する特化カーネルを取得
                kernel = get_cuda_kernel(device_idx, k_name)

                # ctypesの参照生存期間(ライフタイム)をスコープ内で確実にロック
                p_ptr = ctypes.c_void_p(p_cuda.data_ptr())
                g_ptr = ctypes.c_void_p(g_cuda.data_ptr())
                c_lr = ctypes.c_float(lr)
                c_eps = ctypes.c_float(eps)
                c_n = ctypes.c_int(n)

                args = [
                    ctypes.addressof(p_ptr),
                    ctypes.addressof(g_ptr),
                    ctypes.addressof(c_lr),
                    ctypes.addressof(c_eps),
                    ctypes.addressof(c_n)
                ]
                arg_arr = (ctypes.c_void_p * len(args))(*args)

                # 転送先デバイスのカレントストリームを取得してカーネル起動
                stream = torch.cuda.current_stream(target_device).cuda_stream

                res = cuda_driver.cuLaunchKernel(
                    kernel,
                    (n + 255) // 256, 1, 1,  # gridDim
                    256, 1, 1,               # blockDim
                    0,                       # sharedMem
                    ctypes.c_void_p(stream), # stream
                    arg_arr,                 # params
                    None
                )

                if res != 0:
                    raise RuntimeError(f"GPU:{device_idx} 内で {k_name.decode()} の実行に失敗 (コード: {res})")

                # ーーー 3. 同期とホスト側(CPU)への厳密な書戻し ーーー
                if is_cpu_tensor:
                    # カーネルの非同期実行完了を確実に待機し計算のズレを防ぐ
                    torch.cuda.current_stream(target_device).synchronize()
                    # 更新された結果を元のCPUパラメータへ正確に書戻す
                    p.copy_(p_cuda)

        return loss

"""
 https://github.com/muooon/qpola
 True Gradient will guide you through it all; believing in it and continuing to move forward is what fosters growth.
 Don’t let the past control you—the noise within the past is the very source of your worries and suffering.
"""