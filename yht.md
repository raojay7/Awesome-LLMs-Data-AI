# 模型蒸馏
## 蒸馏阶段
### 蒸馏推理阶段
#### [Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision](https://arxiv.org/pdf/2312.09390) Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, Jeff Wu：用教师输出的弱监督信号直接对学生进行微调
#### [Alice: Proactive Learning with Teacher's Demonstrations for Weak-to-Strong Generalization Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/pdf/2504.07316) Shujin Wu1,2∗ Cheng Qian1 Yi R. (May) Fung1 Paul Pu Liang3 Heng Ji1 1University of Illinois Urbana-Champaign 2University of Southern California 3Massachusetts Institute of Technology :学生模型根据教师模型提供的不确定性表达来达到蒸馏效果
## 蒸馏架构
### 跨模型架构蒸馏
#### [Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs](https://openreview.net/pdf?id=bwRxXiGO9A)  Nicolas Boizard,Kevin El Haddad,Céline Hudelot,Pierre Colombo,Equall.ai:直接匹配不同Tokenizers的输出分布的形状
#### [Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation](https://arxiv.org/pdf/2412.14528) Xiao Cui1*, Mo Zhu2*, Yulei Qin3, Liang Xie2,4, Wengang Zhou1, Houqiang Li1:对OT进行扩展到多层级，兼顾了上下文的关系
## 蒸馏效率优化
### 稀疏蒸馏
#### [Sparse Logit Sampling: Accelerating Knowledge Distillation in LLM]（https://arxiv.org/pdf/2503.16870）  Anshumann，MohdAbbasZaidi，Akhil Kedia，Jinwoo Ahn，Taehwak Kwon，KangwookLee，HaejunLee，JoohyungLee:仅对教师模型的Top-K logits进行计算loss
## Agent蒸馏
#### [AGENTDISTILL: TRAINING-FREE AGENT DISTILLATION WITH  GENERALIZABLE MCP BOXES](https://arxiv.org/pdf/2506.14728) Jiahao Qiu, Xinzhe Juan, Yimin Wang, Ling Yang, Xuan Qi, Tongcheng Zhang, Jiacheng Guo,Yifu Lu, Zixin Yao, Hongru Wang, Shilong Liu, Xun Jiang, Liu Leqi, Mengdi Wang:通过直接将教师Agent的策略和知识打包成"MCP Box"，实现无需训练的Agent蒸馏






