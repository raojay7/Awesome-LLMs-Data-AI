Test-time策略
===
解码时优化
---
Test-Time Preference Optimization: On-the-Fly Alignment via Iterative Textual Feedback, Yafu Li, Xuyang Hu, Xiaoye Qu, Linjie Li, Yu Cheng, ICML2025， https://arxiv.org/abs/2501.12895v1, TPO提出了一种在模型使用过程中（即推理时） 实时优化输出以满足偏好的方法，核心特点是无需更新模型参数。通过LLM对于模型生成的输出进行评价，区分好输出和坏输出，再根据二者差异提出修改建议，使LLM refine自己的回答，即通过修改LLM上下文优化输出，但不改变模型参数。



评估时优化
---
### LLM作为验证器替代奖励模型


模型优化
===
Process-based Self-Rewarding Language Models, Shimao Zhang, Xiao Liu, Xin Zhang, Junxiao Liu, Zheheng Luo, Shujian Huang, Yeyun Gong, https://arxiv.org/abs/2503.03746, 基于过程的自奖励模型将自奖励算法的粒度扩展到步骤级，使模型能够评估自己每一个步骤推理步骤的好坏并自己生成奖励信号通过不断迭代优化自己的输出



