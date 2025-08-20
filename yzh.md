## Human-Al collaboration ： reward modeling

### 判别式奖励模型

*The Lessons of Developing Process Reward Models in Mathematical Reasoning, Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin：揭漏了目前主流的蒙特卡洛估计数据标注方法和BON方法的局限性，提出了共识过滤机制（MC和LLM-as-Judge双保险）以及双轨评估机制（答案级评估+步骤级评估），优化了过程奖励模型，在BoN评估和逐步错误识别任务中显著提高了模型性能和数据效率

*VisualPRM: An Effective Process Reward Model for Multimodal Reasoning, Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, Lewei Lu, Haodong Duan, Yu Qiao, Jifeng Dai, Wenhai Wang：提出了多模态过程奖励模型VisualPRM，通过Best-of-N的评估策略可以显著提高MLLM的推理性能

*Inference-Time Scaling for Generalist Reward Modeling, Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, Yu Wu：提出通用奖励模型，通过SPCT方法训练出DeepSeek-GRM，且该通用奖励模型在推理时扩展有更好的性能，为LMM提供更准确的奖励信号

*TLCR: Token-Level Continuous Reward for Fine-grained Reinforcement Learning from Human Feedback, Eunseop Yoon， Hee Suk Yoon， SooHwan Eom， Gunsoo Han， Daniel Wontae Nam， Daejin Jo， Kyoung-Woon On， Mark A. Hasegawa-Johnson， Sungwoong Kim， Chang D. Yoo, ACL2024：提出TLCR奖励模型，为细粒度的PLHF提供Token级的连续奖励

### 生成式奖励模型

*GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning,  Jian Zhao, Runze Liu, Kaiyan Zhang, Zhimu Zhou, Junqi Gao, Dong Li, Jiafei Lyu, Zhouyi Qian, Biqing Qi, Xiu Li and Bowen Zhou：提出生成式过程奖励模型，在对步骤打分之前会先生产分析，再进行验证，最后才对结果进行打分，性能明显优于常规PRM，在数学推理任务取得较大提升

*Generative Verifiers: Reward Modeling as Next-Token Prediction, Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, Rishabh Agarwal, ICLR202：提出生成验证器，利用LLM的文本生成能力来进行验证，优化Best-of-N方法，在算法和数学领域取得了显著的提升。

*RM-R1: Reward Modeling as Reasoning, Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, Hanghang Tong, Heng Ji：提出了新的生成奖励模型，即推理奖励模型，将奖励建模制定为推理任务，能够有效生成明确的评分标准和基本原理链，产生更加一致和有效的奖励信号

*RewardAnything: Generalizable Principle-Following Reward Models, Zhuohao Yu, Jiali Zeng, Weizheng Gu, Yidong Wang, Jindong Wang, Fandong Meng, Jie Zhou, Yue Zhang, Shikun Zhang, Wei Ye：提出原则遵循奖励模型RewardAnything，从隐式偏好学习转向显式原则遵循，可通过自然语言直接控制RM行为，无需收集新数据或重新训练RM

### 隐式奖励模型

*Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning, Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, Aviral Kumar：优化了过程奖励，提出了过程优势验证器PAV方法，即通过奖励进展而非单纯的正确性，以优势函数作为过程奖励，显著提高了计算效率和准确性

*Process Reinforcement through Implicit Rewards, Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Xu Han, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, Ning Ding：提出PRIME框架，在过程奖励模型的基础上提出隐式过程奖励模型，支持奖励模型在线更新，在数学和推理任务中取得显著成果


## Synthetic Feedback (Algorithms adapt to data)

*Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models, Pengyi Li, Matvey Skripkin, Alexander Zubrey, Andrey Kuznetsov, Ivan Oseledets, https://arxiv.org/abs/2506.06395：利用模型自身的信心构建奖励信号，以此来进行强化学习，消除了对标签、奖励模型等的依赖

*Test-Time Preference Optimization: On-the-Fly Alignment via Iterative Textual Feedback, Yafu Li, Xuyang Hu, Xiaoye Qu, Linjie Li, Yu Cheng, ICML2025， https://arxiv.org/abs/2501.12895v1 ：TPO提出了一种在模型使用过程中（即推理时）实时优化输出以满足偏好的方法，核心特点是无需更新模型参数。通过LLM对于模型生成的输出进行评价，区分好输出和坏输出，再根据二者差异提出修改建议，使LLM refine自己的回答，即通过修改LLM上下文优化输出，但不改变模型参数。

*Scalable Best-of-N Selection for Large Language Models via Self-Certainty, Zhewei Kang, Xuandong Zhao, Dawn Song, https://arxiv.org/abs/2502.18581：本文提出了一种名为自确定性的新的置信度指标，通过LLM输出时本身固有的概率分布来估计回答的质量，分布越集中，LLM对其的信心越高，反之越低

*Process-based Self-Rewarding Language Models, Shimao Zhang, Xiao Liu, Xin Zhang, Junxiao Liu, Zheheng Luo, Shujian Huang, Yeyun Gong, https://arxiv.org/abs/2503.03746：基于过程的自奖励模型将自奖励算法的粒度扩展到步骤级，使模型能够评估自己每一个步骤推理步骤的好坏并自己生成奖励信号通过不断迭代优化自己的输出，同时在偏好优化阶段采用DPO，优化了旧方法只打分的局限性，让模型通过比较不同输出来评判好坏。

*Bootstrapping Language Models with DPO Implicit Rewards, Changyu Chen, Zichen Liu, Chao Du, Tianyu Pang, Qian Liu, Arunesh Sinha, Pradeep Varakantham, Min Lin, ICLR 2025, https://arxiv.org/abs/2406.09760：：模型在经过DPO训练后内生了一套评判标准（隐式奖励模型），通过优化该评判标准提出DICE方法，让模型利用这一套标准在不需要额外反馈的条件下来提升自己





# 数据策略提升
## Test-time策略

Bag of Tricks for Inference-time Computation of LLM Reasoning, Fan Liu, Wenshuo Chao, Naiqiang Tan, Hao Liu, https://arxiv.org/abs/2502.07191v4

s1: Simple test-time scaling, Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto, https://arxiv.org/abs/2508.01543

Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering, William Jurayj, Jeffrey Cheng, Benjamin Van Durme, ACL 2025, https://arxiv.org/abs/2502.13962

Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning, Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Xu Han, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, Ning Ding, https://arxiv.org/abs/2502.01456

Scaling Test-Time Compute Without Verification or RL is Suboptimal, Amrith Setlur, Nived Rajaraman, Sergey Levine, Aviral Kumar, https://arxiv.org/abs/2502.12118

Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models, Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, Wanxiang Che, https://arxiv.org/abs/2503.09567

### 解码时优化
Test-Time Preference Optimization: On-the-Fly Alignment via Iterative Textual Feedback, Yafu Li, Xuyang Hu, Xiaoye Qu, Linjie Li, Yu Cheng, ICML2025， https://arxiv.org/abs/2501.12895v1 

TPO提出了一种在模型使用过程中（即推理时） 实时优化输出以满足偏好的方法，核心特点是无需更新模型参数。通过LLM对于模型生成的输出进行评价，区分好输出和坏输出，再根据二者差异提出修改建议，使LLM refine自己的回答，即通过修改LLM上下文优化输出，但不改变模型参数。

Efficient Reasoning Models: A Survey, Sicheng Feng, Gongfan Fang, Xinyin Ma, Xinchao Wang, https://arxiv.org/abs/2504.10903v1

本调查旨在全面概述有效推理的最新进展。它将现有工作分为三个关键方向：（1）更短——将冗长的 CoT 压缩为简洁而有效的推理链;（2）更小——通过知识蒸馏、其他模型压缩技术、强化学习等技术，开发具有强大推理能力的紧凑型语言模型;（3）更快——设计高效的解码策略以加速推理(Test-time策略)。



### 评估时优化
#### 验证方式优化
Scalable Best-of-N Selection for Large Language Models via Self-Certainty, Zhewei Kang, Xuandong Zhao, Dawn Song, https://arxiv.org/abs/2502.18581

本文提出了一种名为自确定性的新的置信度指标来替代奖励模型，通过LLM输出时本身固有的概率分布来估计回答的质量，分布越集中，LLM对其的信心越高，反之越低，本文将这种分布通过统计学方法量化为置信度分数，同时给出了结合Borda投票和自确定性的一种更优的评估方法。同时相比于平均token熵，其更具拓展性，不会受到生成文本长度限制。且能与思维链协同，适用于开放任务领域。

Language Models Prefer What They Know: Relative Confidence Estimation via Confidence Preferences, Vaishnavi Shrivastava, Ananya Kumar, Percy Liang, https://www.arxiv.org/abs/2502.01126

语言模型应该提供可靠的置信度估计，论文提出相对置信度估计的一种方法，通过对比不同问题与回答来选出更优答案，与现有流行的两种绝对置信度估计比较，得出相对置信度估计优于绝对置信度估计，同时探讨了一些方法如思路链对相对置信度估计的影响

#### LLM作为验证器替代奖励模型

Generative Verifiers: Reward Modeling as Next-Token Prediction, Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, Rishabh Agarwal, ICLR2025, https://arxiv.org/abs/2408.15240

基于LLM的验证器通常作为判别模型训练，未利用LLMs的文本生成能力。而简单使用现成LLM作为评判者（LLM-as-a-Judge）的方法在推理任务中表现也不佳。因此本文提出生成验证器，利用LLM的文本生成能力来进行验证，优化Best-of-N方法。核心机制： GenRM 使用LLM最基础、最自然的训练目标——下一个词预测 (Next-Token Prediction) 来执行验证任务。如何表示正确性？ 关键创新在于它不预测一个独立的数值分数。相反，它将解决方案的正确性本身视为一个需要预测的“词元” (Token)。同时，本文给出了直接验证方法，集成了奖励模型和监督微调（通过两种数据微调模型，属于模型优化方面）的方法以及思维链验证方法。

Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers, Shalev Lifshitz, Sheila A. McIlraith, Yilun Du, https://arxiv.org/abs/2502.20379

多验证器提出验证的新维度，采用多个方面验证器（AVS）分别验证输出的不同方面，来判别输出的好坏，并给出了一种简单的分数聚合方法BoN-MAVS。通过验证器模型，验证方面，验证策略三个维度构建验证器工程。



# 模型优化

Process-based Self-Rewarding Language Models, Shimao Zhang, Xiao Liu, Xin Zhang, Junxiao Liu, Zheheng Luo, Shujian Huang, Yeyun Gong, https://arxiv.org/abs/2503.03746

基于过程的自奖励模型将自奖励算法的粒度扩展到步骤级，使模型能够评估自己每一个步骤推理步骤的好坏并自己生成奖励信号通过不断迭代优化自己的输出，同时在偏好优化阶段采用DPO，优化了旧方法只打分的局限性，让模型通过比较不同输出来评判好坏，因为相比打分，模型更擅长作比较。

MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration, Zhitao He, Sandeep Polisetty, Zhiyuan Fan, Yuchen Huang, Shujin Wu, Yi R. Fung, ACL 2025, https://arxiv.org/abs/2505.23224v3

本文提出了一种新的微调框架，通过推理步骤置信度校准来推进 MLLM 的知识边界意识，让MLLM知道对自己输出的每一句话表明置信度。在SFT阶段通过聚合Length-normalized log probability（模型生成这句话的概率平均值（概率越高，信心越高）），Mean token entropy（生成每个词时的“犹豫程度”平均值（越犹豫，信心越低）），TokenSAR（考虑每个词对整句话重要性的加权对数概率（重要词概率高，信心高）），CLIPScore（生成的句子和输入图片的相关性（相关性越高，信心越高））四种指标来生成自己的置信度指标，以对训练数据进行标注，将其分为5个置信度等级，对应不同的MLLM置信度语句池。在强化学习阶段，定义了三个奖励信号，知识准确性奖励（确保模型回答的内容本身是正确的（置信度表达再准，内容错了也没用）），期望校准奖励（让模型表达的置信度能准确反映这句话实际的对错。惩罚过度自信（说很确定但错了）和缺乏自信（说不确定但对了）），置信自校准奖励（让模型表达的置信度与它内部计算的真实信心 (Internal Confidence) 保持一致。鼓励模型“心口如一”。），采用ppo进行优化。

RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning, Kaiwen Zha, Zhengqi Gao, Maohao Shen, Zhang-Wei Hong, Duane S. Boning, Dina Katabi, https://arxiv.org/abs/2505.15034

