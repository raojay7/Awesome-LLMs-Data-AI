#### **[MAGPIE: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)**  
*Zhangchen Xu, Fengqing Jiang, Luyao Niu, et al., 2024*

**类别归属**：3.4. Alignment  Data generation
本论文提出 MAGPIE，一种无需任何人工输入或种子问题的自动化方法，用于从对齐的大语言模型（如 LLaMA-3-Instruct）中合成大规模、高质量的指令数据，以实现模型对齐任务。

**方法核心**：
- **Step 1：Instruction 自生成**：通过输入预定义的 pre-query 模板，利用对齐 LLM 的自回归能力自动生成用户指令。
- **Step 2：Response 自动生成**：将生成的指令作为输入，再由同一模型生成对应回复，构建指令-回复对。
- 支持扩展生成多轮对话（MAGPIE-MT）、偏好数据（MAGPIE-DPO）、领域特定/多语言数据等。

**优势亮点**：
- 不依赖 prompt engineering 或 seed instruction；
- 可扩展、高覆盖、多样性强；
- 自动生成数据在多个对齐基准（AlpacaEval 2, Arena-Hard, WildBench）上优于其他公开数据；
- 用少量（<400K）数据微调出的模型，效果超越官方 LLaMA-3-Instruct（使用 >10M 数据微调）。

**对齐贡献**：
MAGPIE 的核心贡献在于**自动构建对齐数据集**，显著降低了人力成本，同时保证了指令数据的**质量、多样性与安全性**，为大模型的对齐研究提供了一个**可复制、高效的开放方案**，在实际对齐效果上也超过多个现有合成或混合数据集。




#### **[Weak-to-Strong Generalization: Teaching Large Language Models to Self-Debate](https://arxiv.org/abs/2405.14806)**  
*Stephen Casper, Ananth Mahadevan, et al., 2024*

**类别归属**：3.4. Alignment  
**简介**：本论文探讨如何训练大语言模型（LLMs）通过自我辩论（self-debate）来增强对抗性推理能力，实现从“弱者视角”（weaker model’s perspective）进行强泛化（strong generalization）。

**方法概述**：
- 提出**weak-to-strong generalization**框架，鼓励 LLM 学会从较弱模型的视角出发，预测其弱点，并通过与自己“辩论”来推理更强的结论。
- 使用两阶段训练流程：
  - 第一阶段训练模型从弱模型生成的输出中判断正确与否；
  - 第二阶段训练模型进行**自我辩论**，生成正反两个立场的论点并做出更优判断。

**结果与贡献**：
- 提出一种无需直接人类反馈、能够从已有“弱”模型学习更强泛化策略的新思路；
- 实验显示该方法在多个推理数据集（如 AlpacaEval, Arena-Hard）上大幅提升了模型对抗性和辨别力；
- 训练得到的模型更具备 alignment 能力，能够避免接受误导性回答或有害结论。

**对齐贡献**：
这项工作推动了对齐技术从**简单指令跟随**走向**复杂对抗推理与自我监督**，展示了让模型从“自己辩自己”的方式学习更强信念形成机制，为对齐方法（特别是无需人类监督的方向）提供了创新视角和方法论基础。



**Title:** Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs  
**Authors:** Nicolas Boizard, Kevin El Haddad, Céline Hudelot, Pierre Colombo  
**Published:** Transactions on Machine Learning Research (01/2025)

**Category:** 2.2. Distillation

**Summary:**
This paper introduces **Universal Logit Distillation (ULD) loss**, a novel knowledge distillation approach that allows transferring knowledge from a large teacher LLM to a smaller student **across different tokenizers and architectures**. 

Traditional logit-based KD techniques rely on shared token vocabularies, limiting their use between model families. ULD loss overcomes this by using the **Wasserstein distance** to compare logit distributions, avoiding the vocabulary overlap assumption required by KL divergence.

**Key Contributions:**
- Proposes a **tokenizer-agnostic** logit distillation method using **optimal transport** (Wasserstein distance).
- Demonstrates **improved performance** across tasks like extractive QA, generative QA, and summarization.
- Works in **black-box settings**, where the teacher model is frozen and only its logits are used.
- Shows robustness even when using smaller datasets or student models.

**Relevance:** This paper offers a scalable and generalizable solution for **LLM compression via distillation**, particularly useful when the teacher and student models use **incompatible vocabularies or architectures**, making it a clear fit for *2.2. Distillation*.

