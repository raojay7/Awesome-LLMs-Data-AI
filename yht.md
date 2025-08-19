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

# 数学推理
## Evaluation Dataset
- MAWPS [**MAWPS: A Math Word Problem Repository**](https://aclanthology.org/N16-1136.pdf) *Rik Koncel-Kedziorski, Subhro Roy, Aida Amini,Nate Kushman,Hannaneh Hajishirzi* NAACL 2016.
- AQUA-RAT [**Program Induction by Rationale Generation:Learning to Solve and Explain Algebraic Word Problems**](https://arxiv.org/pdf/1705.04146) *Wang Ling, Dani Yogatama, Chris Dyer, Phil Blunsom* IJCAI 2017.
- GSM8K [**Training Verifiers to Solve Math Word Problems**](https://arxiv.org/pdf/2110.14168) *Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman* NeurIPS 2021.
- [**Measuring Mathematical Problem Solving With the MATH Dataset**](https://arxiv.org/abs/2103.03874) *Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt*NeurIPS Dataset and Benchmark 2021.
- [**MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics**](https://arxiv.org/abs/2109.00110) *Kunhao Zheng, Jesse Michael Han, Stanislas Polut*ICLR 2022.
- [**Lila: A Unified Benchmark for Mathematical Reasoning**](https://arxiv.org/abs/2210.17517v2) *Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, Ashwin Kalyana*EMNLP 2022.
- [**ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning**](https://arxiv.org/abs/2203.10244) *Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, Enamul Hoque*ACL(finding) 2022.
- [**UniGeo: Unifying Geometry Logical Reasoning via Reformulating Mathematical Expression**](https://arxiv.org/abs/2212.02746) *Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, Xiaodan Liang*EMNLP 2022.
- [**TheoremQA: A Theorem-driven Question Answering dataset**](https://arxiv.org/abs/2305.12524) *Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, Tony Xia*EMNLP 2023.
- [**CMATH: Can Your Language Model Pass Chinese Elementary School Math Test?**](https://arxiv.org/abs/2306.16636) *Tianwen Wei, Jian Luan, Wei Liu, Shuang Dong, Bin Wang*Preprint 2023.
- [**LeanDojo: Theorem Proving with Retrieval-Augmented Language Models**](https://arxiv.org/abs/2306.15626) *Kaiyu Yang, Aidan M. Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan J. Prenger, Animashree Anandkumar*NeurIPS 2023.
- [**OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems**](https://arxiv.org/abs/2402.14008) *Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, Maosong Sun* ACL 2024.
- [**AIME24**](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) 该数据集包含来自 2024 年美国数学邀请赛 (American Invitational Mathematics Examination, AIME) 的题目。AIME 是一项以其极具挑战性的数学问题而闻名的、享有盛誉的高中数学竞赛。
- [**MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts**](https://arxiv.org/abs/2310.02255) *Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, Jianfeng Gao*ICLR 2024.
- [**MathPile: A Billion-Token-Scale Pretraining Corpus for Math**](https://arxiv.org/abs/2312.17120) *Zengzhi Wang, Xuefeng Li, Rui Xia, Pengfei Liu* NeurIPS 2024.
- [**DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning**](https://arxiv.org/pdf/2504.11456) *Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong Yu*. Preprint 2025
- [**MegaMath: Pushing the Limits of Open Math Corpora**](https://arxiv.org/abs/2504.02807) *Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong YuFan Zhou, Zengzhi Wang, Nikhil Ranjan, Zhoujun Cheng, Liping Tang, Guowei He, Zhengzhong Liu, Eric P. Xing*. Preprint 2025
## Synthetic Dataset
- [**Are NLP Models really able to Solve Simple Math Word Problems?**](https://arxiv.org/pdf/2103.07191) *Arkil Patel, Satwik Bhattamishra, Navin Goyal* NeurIPS 2021.
- [**ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning**](https://arxiv.org/abs/2203.10244) *Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, Enamul Hoque*ACL(finding) 2022.（数据集中既有人类编写问题，也有根据人类编写的图标摘要生成的问题）
- [**Llemma: An Open Language Model For Mathematics**](https://arxiv.org/abs/2310.10631) *Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen Marcus McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, Sean Welleck* ICLR 2024.
- [**MAMMOTH: BUILDING MATH GENERALIST MODELS THROUGH HYBRID INSTRUCTION TUNING**](https://arxiv.org/abs/2309.05653) *Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, Wenhu Chen*ICLR 2024.
- [**RMath: A Logic Reasoning-Focused Datasets Toward Mathematical Multistep Reasoning Tasks**](https://ojs.aaai.org/index.php/AAAI/article/view/34585) *Ziyi Hu, Jun Liu, Zhongzhi Liu, Yuzhong Liu, Zheng Xie, Yiping Song* AAAI 2025.
- [**DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning**](https://arxiv.org/pdf/2504.11456) *Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong Yu*. Preprint 2025（有合成数据补充）
- [**MegaMath: Pushing the Limits of Open Math Corpora**](https://arxiv.org/abs/2504.02807) *Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong YuFan Zhou, Zengzhi Wang, Nikhil Ranjan, Zhoujun Cheng, Liping Tang, Guowei He, Zhengzhong Liu, Eric P. Xing*. Preprint 2025（有合成数据补充）


# 代码生成
- [**CodeSearchNet Challenge: Evaluating the State of Semantic Code Search**](https://arxiv.org/abs/1909.09436) *Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, Marc Brockschmidt*Preprint 2019.
- [**CodeS: Natural Language to Code Repository via Multi-Layer Sketch**](https://arxiv.org/html/2403.16443) *	Daoguang Zan, Ailun Yu, Wei Liu, Dong Chen, Bo Shen, Wei Li, Yafen Yao, Yongshun Gong, Xiaolin Chen, Bei Guan, Zhiguang Yang, Yongji Wang, Qianxiang Wang, Lizhen Cui*Preprint 2024.
- [**CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion**](https://arxiv.org/abs/2310.11248) *Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, Bing Xiang*NeurIPS 2023.
- [**EvoCodeBench: An Evolving Code Generation Benchmark Aligned with Real-World Code Repositories**](https://arxiv.org/abs/2404.00599) *Jia Li, Ge Li, Xuanming Zhang, Yihong Dong, Zhi Jin*Preprint 2024.
- [**RepoFusion: Training Code Models to Understand Your Repository**](https://arxiv.org/abs/2306.10998) *Disha Shrivastava, Denis Kocetkov, Harm de Vries, Dzmitry Bahdanau, Torsten Scholak*Preprint 2023.

