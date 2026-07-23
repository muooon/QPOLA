import os
import ctypes
import sys
import torch
from torch.optim import Optimizer

"""
QPOLA v1.0.4 260723 (Moment-Free) fp8/int8 対応済  ※ CUDA特性のため4bit未対応
QPOLARIS (Quantization n Polar-Aligned Resetting Instant SGD)
量子化に強い、履歴ゼロ、空間協調(極座標･QJL)による自己適応型SGD
QPOLAは従来のオプティマイザよりも大きな学習率(LR)を設定します(最大値として機能します)
低精度･量子化モデルでの学習はLRを下げてください、通常は LR：1e-3 あたりで安定的に進行します(LoRA)
事前学習やフルファインチューンニングにおいては相応しいスケールに落としてください LR：1e-4 程度等(Pre & FT)
(この仕組みは瞬時的な 勾配の分解と再構成 を行います、複次的に VRAM負荷を削減 しました)
usage ／ 使い方
--optimizer_type=optimizer.qpola.QPOLA
Please place qpola.py and qpola_kernel.ptx in the same folder.
"""

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

# デバイスごとの Loaded Module と Function のキャッシュ
CUDA_MODULES = {}
CUDA_KERNELS = {}

def get_cuda_kernel(device_index: int, kernel_name: bytes):
    global CUDA_MODULES, CUDA_KERNELS
    cache_key = (device_index, kernel_name)
    if cache_key in CUDA_KERNELS:
        return CUDA_KERNELS[cache_key]

    torch.cuda.set_device(device_index)

    ## デバイスごとに PTX モジュールを 1度だけロード
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
    def __init__(self, params, lr=1e-2, eps=1e-8, low_vram=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, eps=eps)
        super(QPOLA, self).__init__(params, defaults)
        self.low_vram = low_vram

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

                # デバイスの調停
                if is_cpu_tensor:
                    target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
                    p_cuda = p.to(target_device)
                    g_cuda = g.to(target_device)
                else:
                    target_device = orig_device
                    p_cuda = p
                    g_cuda = g

                # メモリ連続性の保証と｢非連続テンソル｣への対策
                p_was_not_contiguous = not p_cuda.is_contiguous()
                if p_was_not_contiguous:
                    p_cuda = p_cuda.contiguous()
                if not g_cuda.is_contiguous():
                    g_cuda = g_cuda.contiguous()

                n = p_cuda.numel()
                device_idx = target_device.index if target_device.index is not None else 0
                kernel = get_cuda_kernel(device_idx, k_name)

                # cuLaunchKernel 用の引数配列の構築ロジック
                # 各引数のアドレスではなく、値そのものを ctypes オブジェクトとして生成
                p_ptr = ctypes.c_void_p(p_cuda.data_ptr())
                g_ptr = ctypes.c_void_p(g_cuda.data_ptr())
                c_lr = ctypes.c_float(lr)
                c_eps = ctypes.c_float(eps)
                c_n = ctypes.c_int(n)

                # args には各変数のポインタ(アドレス)を直接格納
                args = [
                    ctypes.byref(p_ptr),
                    ctypes.byref(g_ptr),
                    ctypes.byref(c_lr),
                    ctypes.byref(c_eps),
                    ctypes.byref(c_n)
                ]
                # void* args[] に相当するポインタ配列を作成
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

                # GPUかつ非連続だった場合のインプレース書き戻し
                if p_was_not_contiguous and not is_cpu_tensor:
                    p.copy_(p_cuda)

                # CPU配置の場合の同期と書き戻し
                if is_cpu_tensor:
                    torch.cuda.current_stream(target_device).synchronize()
                    p.copy_(p_cuda)

                # 不要になった一時テンソルを明示的に削除してVRAMを解放
                if p_was_not_contiguous or is_cpu_tensor:
                    del p_cuda
                if not g.is_contiguous():
                    del g_cuda

        # 【選択式】 プールされた未使用VRAMキャッシュの完全解放(クリーンアップ) / 中級者以上向け
        # 毎ステップの解放は速度低下しますが 通常：True です(多くの方に学習可能状態を届けるため)
        # VRAMに余裕がある方は初期化時に low_vram=False を設定してください (Falseで高速化)
        if self.low_vram and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss

"""
 https://github.com/muooon/qpola
 True Gradient will guide you through it all; believing in it and continuing to move forward is what fosters growth.
 Don’t let the past control you—the noise within the past is the very source of your worries and suffering.
"""
