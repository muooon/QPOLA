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
QPOLA uses a larger learning rate (LR) than conventional optimizers (it functions as a maximum value).  

*   For low‑precision / quantized models, reduce the LR. Training typically proceeds stably around LR: 1e‑3 (LoRA).  
*   For pre‑training or full fine‑tuning, lower the LR to an appropriate scale such as LR: 1e‑4 (Pre & FT).  

It prioritizes generality, autonomy, and adaptability in pursuit of new paths for optimization, efficiency, and simplicity.  
In its development, we deeply appreciate the insights of those who came before us—and continue to explore new possibilities beyond them.  
