Test-time策略
===
解码时优化
---
Test-Time Preference Optimization: On-the-Fly Alignment via Iterative Textual Feedback, Yafu Li, Xuyang Hu, Xiaoye Qu, Linjie Li, Yu Cheng, ICML2025， https://arxiv.org/abs/2501.12895v1, TPO提出了一种在模型使用过程中（即推理时） 实时优化输出以满足偏好的方法，核心特点是无需更新模型参数。通过LLM对于模型生成的输出进行评价，区分好输出和坏输出，再根据二者差异提出修改建议，使LLM refine自己的回答，即通过修改LLM上下文优化输出，但不改变模型参数。



评估时优化
---
### 验证方式优化
Scalable Best-of-N Selection for Large Language Models via Self-Certainty, Zhewei Kang, Xuandong Zhao, Dawn Song, https://arxiv.org/abs/2502.18581, 本文提出了一种名为自确定性的新的置信度指标来替代奖励模型，通过LLM输出时本身固有的概率分布来估计回答的质量，分布越集中，LLM对其的信心越高，反之越低，本文将这种分布通过统计学方法量化为置信度分数，同时给出了结合Borda投票和自确定性的一种更优的评估方法。同时相比于平均token熵，其更具拓展性，不会受到生成文本长度限制。且能与思维链协同，适用于开放任务领域。

Language Models Prefer What They Know: Relative Confidence Estimation via Confidence Preferences, Vaishnavi Shrivastava, Ananya Kumar, Percy Liang, https://www.arxiv.org/abs/2502.01126, 语言模型应该提供可靠的置信度估计，论文提出相对置信度估计的一种方法，通过对比不同问题与回答来选出更优答案，与现有流行的两种绝对置信度估计比较，得出相对置信度估计优于绝对置信度估计，同时探讨了一些方法如思路链对相对置信度估计的影响
### LLM作为验证器替代奖励模型


模型优化
===
Process-based Self-Rewarding Language Models, Shimao Zhang, Xiao Liu, Xin Zhang, Junxiao Liu, Zheheng Luo, Shujian Huang, Yeyun Gong, https://arxiv.org/abs/2503.03746, 基于过程的自奖励模型将自奖励算法的粒度扩展到步骤级，使模型能够评估自己每一个步骤推理步骤的好坏并自己生成奖励信号通过不断迭代优化自己的输出，同时在偏好优化阶段采用DPO，优化了旧方法只打分的局限性，让模型通过比较不同输出来评判好坏，因为相比打分，模型更擅长作比较。



