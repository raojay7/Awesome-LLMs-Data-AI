# 弱点数据相关论文总结

## 领域型弱点分类
- **[AutoDetect: Automatic Detection of Weaknesses in Large Language Models via Multi-Agent Collaboration](https://arxiv.org/abs/2406.16714) Jiale Cheng, Yida Lu, Xiaotao Gu, Pei Ke, Xiao Liu, Yuxiao Dong, Hongning Wang, Jie Tang, Minlie Huang**：提出多智能体协作框架，通过考官、出题者、评估者迭代生成挑战性任务，自动发现并利用多任务场景下的大模型弱点。
- **[CDS: Knowledge Component-Driven Data Synthesis Guided by Cognitive Diagnosis Theory](https://arxiv.org/abs/2501.07674)  Haokun Zhao, Jinyi Han, Jiaqing Liang, Yanghua Xiao, Xiaojun Meng, Jiansheng Wei**：借鉴认知诊断理论，以知识点掌握情况为导向合成针对性数据，提升模型弱项能力。
- **[SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.08989) Xiao Liang, Zhong-Zhi Li, Yeyun Gong, Yang Wang, Hengyuan Zhang, Yelong Shen, Ying Nian Wu, Weizhu Chen**：在 RL 训练中识别模型反复失败的样本，提取核心概念并合成针对性新题，逐步克服模型弱点。

## 错误类型驱动弱点分类
- **[Error Classification of Large Language Models on Math Word Problems: A Dynamically Adaptive Framework](https://arxiv.org/abs/2501.15581) Yuhong Sun, Zhangyue Yin, Xuanjing Huang, Xipeng Qiu, Hui Zhao**：2025. 构建约 30 万真实数学错误样本数据集，引入动态错误分类与 Error-Aware Prompting，引导模型规避常见推理错误。
- **[Reinforcement Learning on Incorrect Synthetic Data via Advantage-Weighted Reward Shaping for Mathematical Reasoning](https://arxiv.org/abs/2406.14532) Amrith Setlur, Saurabh Garg, Xinyang Geng, Naman Garg, Virginia Smith, Aviral Kumar**：利用负样本和逐步信用分配优化推理关键步骤，实现等效于扩大 8 倍正样本的数学推理效率提升。
- **[Error Typing for Smarter Rewards: Improving Process Reward Models with Error-Aware Hierarchical Supervision (PathFinder-PRM)](https://arxiv.org/abs/2505.19706) Tej Deep Pala, Panshul Sharma, Amir Zadeh, Chuan Li, Soujanya Poria**：在过程奖励模型中引入错误类型检测，将数学与一致性错误分离识别，提升多步骤推理过程的奖励精度。
- **[Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning](https://arxiv.org/abs/2505.22591) Erxin Yu, Jing Li, Ming Liao, Qi Zhu, Boyang Xue, Minghui Xu, Baojun Wang, Lanqing Hong, Fei Mi, Lifeng Shang**：以错误类型为单位聚类 bad cases，并据此生成具泛化性的训练数据，提升数学推理的错误纠正能力。
- **[Evaluating Mathematical Reasoning of Large Language Models: A Focus on Error Identification and Correction](https://arxiv.org/abs/2406.00755) Xiaoyuan Li, Wenjie Wang, Moxin Li, Junrong Guo, Yang Zhang, Fuli Feng**：从阅卷者视角设计四个细粒度错误相关任务（错误存在检测、首错步骤定位、错误类型分类、错误修正），构建包含九类常见错误的数学推理测试集，系统评估模型在不同错误类型和提示下的鲁棒性。
- **[LEMMA: Learning from Errors for MatheMatical Advancement in LLMs](https://arxiv.org/abs/2503.17439) Zhuoshi Pan, Yu Li, Honglin Lin, Qizhi Pei, Zinan Tang, Wei Wu, Chenlin Ming, H. Vicky Zhao, Conghui He, Lijun Wu**：提出从错误中学习的数学推理增强框架，通过教师模型有针对性地制造错误并生成反思-修正数据，结合修正并继续和重启解题两种策略训练模型，实现更强的自主纠错能力。

## 幻觉与上下文忠实性
- **[Teaching with Lies: Curriculum DPO on Synthetic Negatives for Hallucination Detection](https://arxiv.org/abs/2505.17558) Shrey Pandit, Ashwin Vinod, Liu Leqi, Ying Ding**：引入人工构造的高质量幻觉样本作为负样本，并通过课程式 DPO 逐步提升模型的幻觉检测能力。
- **[CANOE: Training Large Language Models to Maintain Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning](https://huggingface.co/papers/2505.16483) Shuzheng Si,Haozhe Zhao,Cheng Gao,Yuzhuo Bai,Zhitong Wang,Bofei Gao,Kangyang Luo,Wenhao Li,Yufei Huang,Gang Chen,Fanchao Qi,Minjia Zhang,Baobao Chang,Maosong Sun**：基于知识三元组自动合成多类型 QA 任务，并通过 Dual-GRPO 优化长短文本的上下文忠实性。
  
## 过程检测数学评测
- **[Exposing the Achilles’ Heel: Evaluating LLMs Ability to Handle Mistakes in Mathematical Reasoning](https://arxiv.org/abs/2406.10834) Joykirat Singh, Akshay Nambi, Vibhav Vineet**：提出了 MWP-MISTAKE 数据集，并通过构造规则错误和小模型错误，系统评测大模型在数学推理中发现与纠正错误的能力，揭示了GPT-4o在纠错上最强但在新题集上仍存在泛化不足及潜在数据污染问题。

## 多智能体交互训练
- **[Multi-agent KTO: Reinforcing Strategic Interactions of Large Language Models in Language Game (MaKTO)](https://arxiv.org/abs/2501.14225) Rong Ye, Yongxin Zhang, Yikai Zhang, Haoyu Kuang, Zhongyu Wei, Peng Sun**：将语言、策略、意图一体化，通过多智能体交互博弈和 KTO 训练，使模型在复杂社交推理游戏中学会战略沟通。

