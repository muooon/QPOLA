import os
import ctypes
import sys
import torch
from torch.optim import Optimizer

"""
QPOLA v1.0.5 (Intel XPU / SYCL版) by muooon
量子化に強い、履歴ゼロ、空間協調(極座標･QJL)、Zero-Master Weight による自己適応型SGD
"""

current_dir = os.path.dirname(os.path.abspath(__file__))

# 拡張子の判定 (Windows は .dll / .pyd、Linux は .so)
if sys.platform.startswith('win'):
    lib_path = os.path.join(current_dir, "qpola_xpu_kernel.dll")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(current_dir, "qpola_xpu_kernel.pyd")
else:
    lib_path = os.path.join(current_dir, "qpola_xpu_kernel.so")

if not os.path.exists(lib_path):
    raise FileNotFoundError(f"QPOLA 共有ライブラリが見つかりません: {lib_path}")

try:
    # SYCL/C++ でビルドした共有ライブラリをロード
    xpu_lib = ctypes.CDLL(lib_path)
except OSError as e:
    raise RuntimeError(f"QPOLA ライブラリのロードに失敗しました: {e}")

class QPOLA(Optimizer):
    def __init__(self, params, 
                 lr=1e-3, 
                 eps=1e-8, 
                 low_vram=True, 
                 betas=(0.9, 0.995), 
                 weight_decay=0.01):
        defaults = dict(lr=lr, eps=eps)
        super(QPOLA, self).__init__(params, defaults)
        self.low_vram = low_vram
        # betas, weight_decay 等は未使用、学習側の記述書き換え等を受け流しエラー防止するダミーです

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
                is_cpu_tensor = not (orig_device.type == 'xpu')

                dtype_str = str(orig_dtype)
                
                # 型に応じたC++側の関数名をマッピング
                if orig_dtype == torch.float32:
                    kernel_func = xpu_lib.qpola_kernel_fp32
                elif orig_dtype == torch.float16:
                    kernel_func = xpu_lib.qpola_kernel_fp16
                elif orig_dtype == torch.bfloat16:
                    kernel_func = xpu_lib.qpola_kernel_bf16
                elif orig_dtype == torch.int8:
                    kernel_func = xpu_lib.qpola_kernel_int8
                else:
                    raise NotImplementedError(f"QPOLAは現在、型 {orig_dtype} をXPU上でサポートしていません")

                # デバイスの調停 (CPU上のテンソルならカレントのXPUへ一時転送)
                if is_cpu_tensor:
                    target_device = torch.device(f"xpu:{torch.xpu.current_device()}")
                    p_xpu = p.to(target_device)
                    g_xpu = g.to(target_device)
                else:
                    target_device = orig_device
                    p_xpu = p
                    g_xpu = g

                # メモリ連続性の保証
                p_was_not_contiguous = not p_xpu.is_contiguous()
                if p_was_not_contiguous:
                    p_xpu = p_xpu.contiguous()
                if not g_xpu.is_contiguous():
                    g_xpu = g_xpu.contiguous()

                n = p_xpu.numel()
                device_idx = target_device.index if target_device.index is not None else 0

                # 引数の型定義 (sycl::queue&, T*, const T*, float, float, int)
                # sycl::queue は内部ポインタとして渡されるため void* でハンドリング
                q = torch.xpu.default_dq(target_device) if hasattr(torch.xpu, 'default_dq') else None
                
                # C++側のエントリーポイント呼び出し
                # 引数シグネチャ: (void* queue_ptr, void* p_ptr, void* g_ptr, float lr, float eps, int n)
                # ※C++側の実装に合わせて呼び出し形式を合わせています
                kernel_func.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_float,
                    ctypes.c_float,
                    ctypes.c_int
                ]
                
                # PyTorchのXPUストリーム/キューのポインタを取得(互換用のポインタ取得)
                # PyTorch for XPUのバージョンに合わせたSYCLキューポインタの取得
                stream = torch.xpu.current_stream(target_device)
                # 利用可能な属性に応じてSYCLキューのポインタを取り出す
                queue_ptr = getattr(stream, 'sycl_queue', None)
                if queue_ptr is None and hasattr(stream, 'queue'):
                    queue_ptr = stream.queue
                stream_ptr = ctypes.c_void_p(queue_ptr)

                p_ptr = ctypes.c_void_p(p_xpu.data_ptr())
                g_ptr = ctypes.c_void_p(g_xpu.data_ptr())

                # カーネル実行
                kernel_func(
                    stream_ptr,
                    p_ptr,
                    g_ptr,
                    ctypes.c_float(lr),
                    ctypes.c_float(eps),
                    ctypes.c_int(n)
                )

                # 非連続だった場合のインプレース書き戻し
                if p_was_not_contiguous and not is_cpu_tensor:
                    p.copy_(p_xpu)

                # CPU配置の場合の同期と書き戻し
                if is_cpu_tensor:
                    torch.xpu.synchronize(target_device)
                    p.copy_(p_xpu)

                # 一時テンソルの解放
                if p_was_not_contiguous or is_cpu_tensor:
                    del p_xpu
                if not g.is_contiguous():
                    del g_xpu

        # VRAMキャッシュの解放制御
        if self.low_vram and torch.xpu.is_available():
            torch.xpu.empty_cache()

        return loss

"""
 https://github.com/muooon/qpola
 True Gradient will guide you through it all; believing in it and continuing to move forward is what fosters growth.
"""