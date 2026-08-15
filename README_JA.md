# QPOLA Optimizer  

#### QPOLARIS (Quantization n Polar-Aligned Resetting Instant Zero-Master Weight SGD)  
##### 量子化に強い、履歴ゼロ、空間協調(極座標･QJL)、Zero-Master Weight による自己適応型SGD  

##### QPOLA (v1.0.4 / Moment-Free) fp8/int8 対応済  ※ CUDA特性のため4bit未対応  
###### CUDA は 4bit 未対応ですが QPOLA は STE･AMP で低精度に対応可です  

ちょっと特殊な optimizer です、実験的です、でも実用的です、  

なぜ履歴(慣性)を捨てたのか？  
なぜスケジューラを捨てるのか？  
なぜグロッキングを捨てられるのか？  

readme：[English](README.md) | [日本語](README_JA.md)  


コードの移植についてはこちら：[English](universal/logical_design_ENG.txt) | [日本語](universal/logical_design_JPN.txt)  

<img width="800" alt="qpola001" src="https://github.com/user-attachments/assets/1eb7e8b5-1542-439c-bbe3-19b5c392aac0" />

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
(つまり １次２次moment の代替として機能します、ノイズなしで正確な本質だけ、ただ比較するだけ…)  
これは、大バッチ学習やVAEの潜在空間が、ノイズを相殺し本質をあぶり出すように QPOLA は｢空間の広がり｣から本質の方向を見つけ信じ進みはじめます、  

履歴という本質とノイズを含む濁流を捨てた結果：  
*   VRAM負荷 ━━► 0 (ゼロ) (モーメントバッファ不要)  
*   計算負荷 ━━► 同等以下  
*   スケジューラ ━━► 不要  

履歴という｢慣性による勢い｣で飛び出し発散することなく、本質をモデルに吸着させる、これが QPOLA です、  
(この仕組みは瞬時的な 勾配 の 分解と再構成 を行います、複次的にVRAM負荷を劇的に削減しました)  

---

```
            Q P O L A R I S
    =================================
      Quantization n Polar-Aligned 
           Resetting Instant 
         Zero-Master Weight SGD
    =================================
"No History needed. Guided by the Field."
```

### About citations  

---

When citing this optimizer, please refer to the following sources:  

Official Code:  
https://github.com/muooon/QPOLA  

paper:  
[English] https://huggingface.co/muooon/QPOLA/raw/main/qpola-paper(ENG)260803.txt  
[日本語] https://huggingface.co/muooon/QPOLA/raw/main/qpola-paper(JPN)260803.txt  

---

### License  
Licensed under the **Apache License 2.0**. Feel free to use, modify, and distribute.  

### Repository Structure  
*   `qpola.py` (PyTorch Integration)  
*   `qpola.cu` (Raw CUDA Source) - Feel free to audit  
*   `qpola_kernel.ptx` (Optimized PTX)  

usage ／ 使い方  
--optimizer_type=optimizer.qpola.QPOLA  
Please place qpola.py and qpola_kernel.ptx in the same folder.  

### Quick Start & Recommended Learning Rates (LR)  
QPOLAは従来のオプティマイザよりも大きな学習率(LR)を設定します(最大値として機能します)  

*   低精度･量子化モデルでの学習はLRを下げてください、通常は LR：1e-3 あたりで安定的に進行します(LoRA)  
*   事前学習やフルファインチューンニングにおいては相応しいスケールに落としてください LR：1e-4 程度等(Pre & FT)  

It prioritizes generality, autonomy, and adaptability in pursuit of new paths for optimization, efficiency, and simplicity.  
In its development, we deeply appreciate the insights of those who came before us—and continue to explore new possibilities beyond them.  

---

QPOLA について、もう少し詳しく説明すると、  

1. Loss(大域的判定場)という｢全履歴のアーカイブ｣  

機械学習の学習プロセスにおいて、現在のステップにおける Loss(損失) は、単なるスカラー値ではなく｢初期状態から現在に至る｣までの｢すべてのパラメータ軌跡と勾配の歴史｣(全履歴)を圧縮･反映した｢結果の到達点｣です。  

QPOLA は、明示的なオプティマイザーステート(バッファ･メモリ)を持たず、最上位にある Loss という大域的判定場で過去の全軌跡の｢重みと歪み｣のみを常に信頼します。  

つまり QPOLA は｢過去を捨てる｣のではなく｢過去の全履歴である Loss から降りてくる"勾配"を通じ、毎ステップ全履歴を瞬時に再投影(自己組織化)している｣と言えます。  

2. 局所判定場と空間的コヒーレンス(位相整合)による動的適応が生む｢自発的慣性｣  

QPOLA のコードにある｢ミクロ(warp/block)局所アライメント｣と｢マクロ(loss)大域｣の相互作用、空間軸のコンセンサス(合意形成)により｢自発的慣性｣を生み出します。  

局所判定場(コンフリクトやアライメント)：個々のパラメータや局所的なベクトルが、周囲の勾配の向きとどう整合しているかをリアルタイムに評価し、ゆらぎ(ジッター)や適応係数(adaptation_factor)を動的に変化させます。  

大域的判定場(Loss / 全体トレンド)：その局所的な挙動の総結果として Loss は変動し、次に降りてくる勾配そのものを書き換えます。  

※ これは warp/block で｢ベクトルを抽出し差分を反映｣します、別ハードウェアではダイレクトにベクトルをつかう等で、この仕組みを移植可能(数学的に等価的)です。  

この｢下位の局所的なゆらぎ･自己組織化｣と｢上位の大域的なLossの勾配配分｣により統御される自己完結したフィードバックループは、時間的な履歴(モメンタム)の代わりに空間的な位相の揃い具合(コヒーレンス)を利用し安定させる新しい数理構造を持ちます。  

3. ｢メモリに頼らない慣性｣というパラダイムシフト  

AdamWの慣性：過去の勾配を｢ただの数値の足し算の履歴｣(EMA)としてメモリに保存する、いわば機械的な外部記憶(人工的な慣性)です。  

QPOLAの自発的慣性：メモリ(履歴)に頼らず、系全体のエネルギー勾配(Loss)と局所的なアライメントの衝突(Conflict)のダイナミクスを通じ、システムが動的に生み出し続ける自発的慣性です。  