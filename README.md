# QPOLA Optimizer  

### QPOLARIS (Quantization n Polar-Aligned Resetting Instant Zero-Master Weight SGD)  
#### Quantization‑resilient, history‑free, spatially coordinated (polar coordinates / QJL) self‑adaptive Zero-Master Weight SGD  

#### QPOLA (v1.0.4 / Moment-Free) fp8/int8 supported  ※ 4‑bit unsupported due to CUDA characteristics  
##### CUDA itself does not support 4‑bit, but QPOLA can still operate in low precision via STE and AMP  

A somewhat unusual optimizer — experimental, yet practical.  

Why did we abandon the history (inertia) ?  
Why are we abandon the scheduler ?  
Why can we abandon Glocking ?  

readme：[English](README.md) | [日本語](README_JA.md)  


For code porting, please check here：[English](https://huggingface.co/muooon/QPOLA/raw/main/universal/logical_design_ENG.txt) | [日本語](https://huggingface.co/muooon/QPOLA/raw/main/universal/logical_design_JPN.txt)  

CANN-Ascend prototype：[code](https://github.com/muooon/QPOLA/tree/main/universal)  


<img width="800" alt="qpola001" src="https://github.com/user-attachments/assets/1eb7e8b5-1542-439c-bbe3-19b5c392aac0" />

Why? Why does abandoning history (1st/2nd moments) and schedulers allow grokking to be avoided?  

First, let’s look directly at grokking (delayed generalization).  
Near the final stage of training, we lower the learning rate (LR) to the extreme using a scheduler in order to “remove noise and stabilize what has been learned.”  
Why? Because the accumulated history—the “unstoppable inertia”—requires a forced brake.  
Due to this extremely low LR, the model loses the power (step size) needed to clean out noise, and stagnation (grokking) emerges, requiring long periods of time.  
This is likely the true nature of the grokking phenomenon.  

In other words, if we never have “inertia” (history) in the first place, schedulers and grokking both disappear.  
The belief that “learning cannot progress without inertia” is probably just an assumption.  

So then, how do we make learning progress without inertia?  

QPOLA uses spatial coordination instead of temporal history (past gradients).  
In parameter space, it compares the GPU hardware hierarchy—“micro” (32/Warp) and “macro” (256/Block)—in real time.  
By comparing macro and micro alignment alone, gradients pointing in “different directions” (noise) cancel each other out, while only the “consistent direction” (the essential signal) automatically emerges.  
(This functions as a substitute for 1st and 2nd moments: no noise, only the true essential direction, simply by comparing spatial structure…)  
Just as large‑batch training or VAE latent spaces cancel noise and reveal underlying structure, QPOLA finds and follows the essential direction from the “spatial extent” of the gradient field.  

By discarding the turbulent stream of history—which contains both essence and noise—the results are:  
*   VRAM load ━━► 0 (zero) (no moment buffers required)  
*   Compute load ━━► equal or lower  
*   Scheduler ━━► unnecessary  

Without “inertial momentum” from historical accumulation, parameters no longer overshoot or diverge; instead, the essential signal adheres directly to the model. This is QPOLA.  
(This mechanism performs instantaneous decomposition and reconstruction of gradients, and as a secondary effect dramatically reduces VRAM usage.)  

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
[English] https://huggingface.co/muooon/QPOLA/raw/main/qpola-paper(ENG).txt  
[日本語] https://huggingface.co/muooon/QPOLA/raw/main/qpola-paper(JPN).txt  

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
QPOLA uses a larger learning rate (LR) than conventional optimizers (it functions as a maximum value).  

*   For low‑precision / quantized models, reduce the LR. Training typically proceeds stably around LR: 1e‑3 (LoRA).  
*   For pre‑training or full fine‑tuning, lower the LR to an appropriate scale such as LR: 1e‑4 (Pre & FT).  

It prioritizes generality, autonomy, and adaptability in pursuit of new paths for optimization, efficiency, and simplicity.  
In its development, we deeply appreciate the insights of those who came before us—and continue to explore new possibilities beyond them.  

---

To explain QPOLA in a bit more detail:  

1. Loss (Global Judgment Field) as an "Archive of All History"  

In the machine learning training process, the Loss at the current step is not merely a scalar value; it is the "destination of results" that compresses and reflects the "entire parameter trajectory and gradient history" (the complete history) from the "initial state up to the present."  

QPOLA does not have explicit optimizer states (buffer memory); instead, it always trusts only the "weights and distortions" of the entire past trajectory through the top-level global judgment field known as Loss.  

In other words, rather than "discarding the past," QPOLA can be described as "instantly re-projecting (self-organizing) the entire history at every step through the 'gradients' descending from Loss, which is the complete past history."  

2. "Spontaneous Inertia" Generated by Dynamic Adaptation via Local Judgment Fields and Spatial Coherence (Phase Synchronization)  

The interaction between the "micro (warp/block) local alignment" and "macro (loss) global" levels in the QPOLA code, along with spatial-axis consensus (consensus building), generates "spontaneous inertia."  

Local Judgment Fields (Conflicts and Alignment): Evaluates in real-time how individual parameters or local vectors align with the direction of surrounding gradients, dynamically altering fluctuations (jitter) and the adaptation factor.  

Global Judgment Field (Loss / Overall Trend): Loss fluctuates as the aggregate result of these local behaviors, rewriting the very gradients that descend next.  

Note: This mechanism extracts vectors and reflects differences using warps/blocks, and can be ported (mathematically equivalent) to other hardware by directly utilizing vectors, etc.  

This self-contained feedback loop—governed by "lower-level local fluctuations and self-organization" and "upper-level global Loss gradient allocation"—possesses a novel mathematical structure that stabilizes systems by utilizing spatial phase alignment (coherence) instead of temporal history (momentum).  

3. A Paradigm Shift: "Inertia Independent of Memory"  

AdamW's Inertia: Saves past gradients in memory as "merely a history of numerical additions" (EMA), which is essentially an external mechanical storage (artificial inertia).  

QPOLA's Spontaneous Inertia: A spontaneous inertia continuously generated dynamically by the system through the dynamics of the overall system's energy gradients (Loss) and local alignment conflicts, without relying on memory (history).  