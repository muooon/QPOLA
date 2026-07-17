# QPOLA Optimizer  

### QPOLARIS (Quantization n Polar-Aligned Resetting Instant SGD)  
#### 量子化に強い、履歴ゼロ、空間協調(極座標･QJL)による自己適応型SGD  

#### QPOLA (v1.0.1 / Moment-Free) fp8/int8 対応済  ※ CUDA特性のため4bit未対応  
ちょっと特殊な optimizer です、実験的です、でも実用的です、  

なぜ履歴(慣性)を捨てたのか？  Why did we abandon the history (inertia) ?  
なぜスケジューラを捨てるのか？  Why are we abandoning the scheduler ?  
なぜグロッキングを捨てられるのか？  Why can we abandon Glocking ?  

なぜ？ 履歴(1st/2nd Moment)、スケジューラ、を捨てることで、グロッキングを回避できるのか？  

まずグロッキング(遅延汎化)を見つめてみましょう、  
学習の最終盤、私たちは｢ノイズを除去し学習内容を定着させる｣ため、スケジューラでLR(学習率)を極限まで下げていきます、  
これはなぜ？ 溜め込んだ履歴という｢止められない"慣性"｣に強制的なブレーキが必要だからです、  
この低LRのせいでノイズを掃除するパワー(歩幅)を失い、停滞(グロッキング)という長い時間を必要とします、  
これが恐らくグロッキング現象の正体です、  

つまり最初から｢慣性｣(履歴)を持たなければ、スケジューラも、グロッキングもなくなる、  
｢慣性なければ学習は進まない｣というのは、恐らくただの思い込みです、  

では、どうやって｢慣性なし｣で学習を進ませるのか？  

QPOLA は、時間軸の履歴(過去の勾配)の代わりに、空間の協調を使います、  
パラメータ空間において、GPUのハードウェア階層である｢ミクロ｣(32/Warp)と｢マクロ｣(256/Block)の差分をリアルタイムに比較するのです、  
このマクロとミクロの比較だけで｢バラバラな方向｣を向いた｢ノイズ｣は互いに打ち消し合い、｢同じ方向｣に一貫して流れる｢本質｣(差分)だけ自動的に浮かび上がります、  
(つまり １次２次moment の代替として機能します、ノイズなしで正確な本質だけ、ただ比較するだけで…)  
これは、大バッチ学習やVAEの潜在空間が、ノイズを相殺し本質をあぶり出すように QPOLA は｢空間の広がり｣から本質の方向を見つけ信じ進みはじめます、  

履歴という本質とノイズを含む濁流を捨てた結果：  
*   VRAM負荷 ━━► 0 (ゼロ) (モーメントバッファ不要)  
*   計算負荷 ━━► 同等以下  
*   スケジューラ ━━► 不要  

履歴という｢慣性による勢い｣で飛び出し発散することなく、本質をモデルに吸着させる、これが QPOLA です、  

---

### License  
Licensed under the **Apache License 2.0**. Feel free to use, modify, and distribute.  

### Repository Structure  
*   `qpola.py` (PyTorch Integration)  
*   `qpola_kernel.cu` (Raw CUDA Source) - Feel free to audit  
*   `qpola_kernel.ptx` (Optimized PTX)  

usage ／ 使い方  
--optimizer_type=optimizer.qpola.QPOLA  
Please place qpola.py and qpola_kernel.ptx in the same folder.  

### Quick Start & Recommended Learning Rates (LR)  
QPOLAは従来のオプティマイザよりも大きな学習率(LR)を設定し高速かつシャープに収束します、  

*   **Transformer** Full Fine-Tuning (FT)：1e-3(1e-2 〜 1e-4) LoRA：1e-2(〜1e-4)  
*   **SDXL / Diffusion** Fine-Tuning (FT)：1e-2(1e-1 〜 1e-3) LoRA：1e-1(〜1e-3)  

It prioritizes generality, autonomy, and adaptability in pursuit of new paths for optimization, efficiency, and simplicity.  
In its development, we deeply appreciate the insights of those who came before us—and continue to explore new possibilities beyond them.  
