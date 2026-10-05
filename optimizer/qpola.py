import torch
from torch.optim import Optimizer

"""
QPOLA v1.1.0 Universal Edition 261005 (PyTorch版･Moment-Free) fp8/int8 対応済
(Pure PyTorch & Strict Layer & Padding-Guard, Cross-Device, AMP Supported)
Quantization n Polar-Aligned Resetting Instant Zero-Master Weight SGD
量子化に強い、履歴ゼロ、空間協調(極座標･QJL)、Zero-Master Weight による自己適応型SGD
QPOLAは従来のオプティマイザよりも大きな学習率(LR)を設定します(最大値として機能します)
低精度･量子化モデルの学習はLRを下げてください、通常は LR：1e-4 程度で安定的に進行します(LoRA/PreTrain)
フルファインチューンニングにおいては相応しいスケールに落としてください LR：1e-6 程度等(FT/FullRank)
事前学習では｢切断正規分布｣による初期化を検討してください
(この仕組みは瞬時的な 勾配の分解と再構成 を行います、複次的に VRAM負荷を削減 しました)
usage ／ 使い方
--optimizer_type=optimizer.qpola.QPOLA
not CUDA Kernel, not PTX Code, not Hardware-specific. 
既存オプティマイザの init に合わせることで未指定項目によるエラーを防止(未使用項目はダミーになります)
"""

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
        # betas, weight_decay 等は既存の学習設定ファイルとの互換用のダミーです

    @torch.no_grad()
    def step(self, closure=None):
        loss = torch.enable_grad()(closure)() if closure is not None else None

        for group in self.param_groups:
            base_lr = group['lr']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                device = p.device
                dtype = p.dtype

                # デバイスまたぎ (CPUオフロード等) の調停
                # 計算は勾配が存在するデバイス (基本はアクセラレータ側) で実行する
                target_device = g.device
                p_work = p if p.device == target_device else p.to(target_device, non_blocking=True)
                g_work = g if g.device == target_device else g.to(target_device, non_blocking=True)

                # メモリ連続性の保証と非連続テンソルへの対策
                p_was_not_contiguous = not p_work.is_contiguous()
                p_target = p_work.contiguous() if p_was_not_contiguous else p_work
                g_target = g_work.contiguous() if not g_work.is_contiguous() else g_work

                # 型ごとのパラメータ (TypeTraits 相当) の設定
                if dtype == torch.float32:
                    clamp_max = 3.4e38
                    min_factor = 1e-5
                    lim_g_hat = 16.0
                    use_stochastic_rounding = False
                    lsb_step = 0.0
                elif dtype == torch.float16:
                    clamp_max = 65504.0
                    min_factor = 1e-4
                    lim_g_hat = 8.0
                    use_stochastic_rounding = False
                    lsb_step = 0.0
                elif dtype == torch.bfloat16:
                    clamp_max = 3.4e38
                    min_factor = 1e-5
                    lim_g_hat = 8.0
                    use_stochastic_rounding = False
                    lsb_step = 0.0
                elif dtype == torch.int8:
                    clamp_max = 127.0
                    min_factor = 1e-2
                    lim_g_hat = 2.0
                    use_stochastic_rounding = True
                    lsb_step = 1.0
                elif "e4m3" in str(dtype):
                    clamp_max = 448.0
                    min_factor = 1e-2
                    lim_g_hat = 4.0
                    use_stochastic_rounding = True
                    lsb_step = 0.0625
                elif "e5m2" in str(dtype):
                    clamp_max = 57344.0
                    min_factor = 1e-2
                    lim_g_hat = 4.0
                    use_stochastic_rounding = True
                    lsb_step = 0.25
                else:
                    clamp_max = 3.4e38
                    min_factor = 1e-5
                    lim_g_hat = 16.0
                    use_stochastic_rounding = False
                    lsb_step = 0.0

                # マスターウェイトをその量子化型のまま、計算時のみfloat化
                p_val = p_target.float()
                g_val = g_target.float()

                # NaN / Inf ガード
                g_val = torch.nan_to_num(g_val, nan=0.0, posinf=0.0, neginf=0.0)
                p_val = torch.nan_to_num(p_val, nan=0.0, posinf=0.0, neginf=0.0)

                # ゼロパディング (有効ではない要素) を計算に含めないためのマスク処理
                is_active = (g_val != 0.0)
                active_count = is_active.sum().clamp(min=1.0).float()

                # 勾配の方向 (符号) と絶対値
                g_sign = torch.sign(g_val)
                g_abs = torch.abs(g_val)

                # レイヤー等をまたがない独立した空間集計(0パディング除外)
                micro_direction_sum = (g_sign * is_active.float()).sum()
                warp_g_scale_sum = (g_abs * is_active.float()).sum()

                micro_direction_mean = micro_direction_sum / active_count
                warp_g_scale = warp_g_scale_sum / active_count

                # 空間アライメント (一致度) / Conflict (衝突度) の算出
                micro_align = g_sign * micro_direction_mean
                diff_micro = torch.clamp(1.0 - micro_align, min=0.0)
                conflict = diff_micro * is_active.float()

                # 減衰係数の算出
                decay_rate = (1.0 - min_factor) * 0.5
                raw_adaptation = 1.0 - conflict * decay_rate
                adaptation_factor = torch.clamp(raw_adaptation, min=min_factor, max=1.0)

                # 勾配の無次元化と飽和 (tanhによるリミッター)
                g_hat_raw = g_val / (warp_g_scale + eps)
                g_hat = torch.tanh(g_hat_raw / lim_g_hat) * lim_g_hat

                # パラメータの更新
                next_p = p_val - (base_lr * g_hat * adaptation_factor)

                # 確率的丸め (Stochastic Rounding) の適用 (低ビット型用)
                if use_stochastic_rounding and lsb_step > 0.0:
                    rand_noise = torch.rand_like(next_p) - 0.5
                    jitter_scale = lsb_step * 0.25 * (1.0 + 0.2 * conflict)
                    next_p = next_p + rand_noise * jitter_scale

                # 値のクランプとガード
                next_p = torch.clamp(next_p, min=-clamp_max, max=clamp_max)
                next_p = torch.where(torch.isnan(next_p) | torch.isinf(next_p), p_val, next_p)

                # 元のテンソル型 (dtype) へキャストしてワーク用テンソルに書き戻し
                p_target.copy_(next_p.to(dtype))

                # 非連続だった場合のインプレース書き戻し
                if p_was_not_contiguous:
                    p_work.copy_(p_target)

                # 元のデバイス (CPU等へのオフロード配置) へ安全に書き戻し
                if p.device != p_work.device:
                    p.copy_(p_work.to(p.device, non_blocking=True))
                    if p.device.type == 'cpu' and torch.cuda.is_available():
                        torch.cuda.synchronize()

                # 不要になった一時テンソルの明示的な削除
                if p_was_not_contiguous:
                    del p_target
                if p.device != p_work.device:
                    del p_work
                if g.device != g_target.device:
                    del g_target

        # [選択式] プールされた未使用VRAMキャッシュの完全解放 (クリーンアップ)
        if self.low_vram and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss

"""
 https://github.com/muooon/qpola
 True Gradient will guide you through it all; believing in it and continuing to move forward is what fosters growth.
 Don’t let the past control you—the noise within the past is the very source of your worries and suffering.
"""
