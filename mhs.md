# self-generation
核心思想是利用大型语言模型自身的能力，通过迭代、演化或自我反馈等机制，来创造新的、高质量的训练数据，从而减少对人类标注的依赖，并实现模型能力的自我提升。
## 自我迭代与博弈
这类方法的核心是通过一个迭代循环来提升模型能力。模型首先生成问题的解决方案，然后通过某种方式（自我评估或对抗）来验证或筛选出高质量的生成结果，并用这些高质量结果对自身进行微调，从而在下一轮迭代中表现得更好
- [**STaR: Bootstrapping Reasoning With Reasoning**](https://arxiv.org/abs/2203.14465) *Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman.* NeurIPS 2022.<br>每一轮迭代输入问题，让模型生成推理过程，如果答案错误则重新生成推理，最终利用所有成功的推理过程来微调自身，从而用推理能力来引导推理能力的提升。

- [**Large Language Models Can Self-Improve**](https://aclanthology.org/2023.emnlp-main.67/) *Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, Jiawei Han.* EMNLP 2023.<br>提出了一个自我提升框架，即让LLM作为学生回答问题，再作为老师对自己的回答提供反馈和示范，然后综合这些反馈来微调自己。

- [**Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models**](https://arxiv.org/abs/2401.01335) *Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu.* ICML 2024.<br>提出 SPIN 方法，让模型与过去的自己进行博弈，将上一轮迭代的LLM作为对手玩家生成尽可能接近人类的合成数据，而这一轮迭代LLM作为主要玩家被训练以区分这些数据，最后通过主要玩家的评估更新对手玩家，从而在不需要更强外部模型的情况下实现自我提升。

- [**Self-playing Adversarial Language Game Enhances LLM Reasoning**](https://arxiv.org/abs/2404.10642) *Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du.* NeurIPS 2024.<br>设计了一个对抗性的语言游戏，其中生成器提出问题和答案，鉴别器判断答案是否可被轻易猜到，通过这种博弈来激励生成器产出更需要复杂推理才能解决的问题。

*****
- [**Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models**](https://arxiv.org/abs/2406.13542) *Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, Jingren Zhou.* ICLR 2025.<br>提出了一个两阶段指令合成框架，以少量手写种子指令为起点，用 LLM Self-Instruct生成指令，并通过交叉验证和回译过滤得到一批可代码验证的高质量原子指令，随后将这些指令与真实用户查询拼接，再次用模型生成并筛选出既符合指令又贴合查询的成对训练样本，形成高质量数据集。

- [**I-SHEEP: Self-Alignment of LLM from Scratch through an Iterative Self-Enhancement Paradigm**](https://arxiv.org/abs/2408.08072) *Yiming Liang, Ge Zhang, Xingwei Qu, Tianyu Zheng, Jiawei Guo, Xinrun Du, Zhenzhu Yang, Jiaheng Liu, Chenghua Lin, Lei Ma, Wenhao Huang, Jiajun Zhang.* ICLR 2025.<br>通过自我合成-自我评估-过滤-训练的迭代循环，让当前模型对少量种子做 in-context重写生成新prompt，再让模型零样本补全回答，形成指令-输出对；每轮后用模型自我评分并只保留高分对，作为下一轮微调与再生成的种子。


## 数据蒸馏
这类方法采用教师-学生范式。它们使用一个更强大的教师LLM来生成一个合成的、带标签的训练数据集。然后，这个数据集被用来训练一个更小的、更高效的学生模型，从而将教师模型在特定任务上的知识和能力迁移给学生模型。
- [**Generating Training Data with Language Models: Towards Zero-Shot Language Understanding**](https://arxiv.org/abs/2202.04538) *Yu Meng, Jiaxin Huang, Yu Zhang, Jiawei Han.* NeurIPS 2022.<br>利用一个已微调的LLM为某个特定任务（如NLI）生成大量带标签的训练样本，然后用这些合成数据去训练一个更小的模型，使其在该任务上达到很好的性能。

- [**ZeroGen: Efficient Zero-shot Learning via Dataset Generation**](https://arxiv.org/abs/2202.07922) *Jiacheng Ye, Jiahui Gao, Qintong Li, Hang Xu, Jiangtao Feng, Zhiyong Wu, Tao Yu, Lingpeng Kong.* EMNLP 2022.<br>设计高效提示让预训练模型（PLM）自动合成下游任务数据并用快速过滤提升质量，并在其之上训练小型任务模型（TAM）以获得零样本学习能力。

- [**Symbolic Knowledge Distillation: from General Language Models to Commonsense Models**](https://arxiv.org/abs/2110.07178) *Peter West, Chandra Bhagavatula, Jack Hessel, Jena D. Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, Yejin Choi.* NAACL 2022.<br>提出了一种更高级的数据蒸馏方式。它不仅仅是蒸馏出最终的（输入，输出）对，而是让教师模型生成中间的符号知识（如常识推理步骤），然后将这些富含逻辑和知识的中间步骤作为训练数据的一部分，来训练学生模型，从而更有效地蒸馏出常识知识

## 指令演化
这类方法的核心在于对指令本身进行优化和创造。它们从一个初始的、可能很简单的指令集出发，通过类似生物进化的变异和选择过程，自动地生成更复杂、更多样、更高质量的新指令，从而构建出强大的指令微调数据集。

- [**Automatic Instruction Evolving for Large Language Models**](https://arxiv.org/abs/2406.00770) *Weihao Zeng, Can Xu, Yingxiu Zhao, Jian-Guang Lou, Weizhu Chen.* EMNLP 2024.<br>借鉴进化计算的思想，自动地将简单、低质量的指令进化成更复杂、高质量的指令。通过多轮的指令生成、变异、筛选和模型再训练，可以自动地创造出一个多样且高质量的指令数据集。

## 多智能体
这类方法利用多个智能体之间的交互来生成数据。

- [**CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society**](https://arxiv.org/abs/2303.17760) *Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, Bernard Ghanem.* NeurIPS 2023.<br>提出了一个新颖的沟通式智能体框架，让一个AI用户和一个AI助手通过角色扮演进行对话来完成任务，从而大规模地生成能反映合作行为和指令遵循能力的对话数据。



## 自生成奖励与偏好
这类方法让LLM自己充当裁判或奖励模型，来为生成的内容打分或提供偏好判断。

- [**Self-Rewarding Language Models.**](https://arxiv.org/abs/2401.10020) *Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston.* ICML 2024.<br>提出一个训练框架，其中LLM在作为奖励模型为自己生成的多个响应进行打分后，再利用这些自我生成的奖励通过DPO等算法进行微调，实现了自给自足的对齐学习。

****
- [**Spread Preference Annotation: Direct Preference Judgment for Efficient LLM Alignment**](https://arxiv.org/abs/2406.04412v2) *Dongyoung Kim, Kimin Lee, Jinwoo Shin, Jaehyung Kim.* ICLR 2025.<br>用极少量人工偏好数据作为种子，通过迭自采样-自打分-自精炼三步，直接利用当前模型 logits 生成偏好标签并去噪，再用 DPO 微调，持续扩散人类偏好先验，实现低成本高效对齐。

- [**Self-Boosting Large Language Models with Synthetic Preference Data**](https://arxiv.org/abs/2410.06961) *Qingxiu Dong, Li Dong, Xingxing Zhang, Zhifang Sui, Furu Wei.* ICLR 2025.<br>提出了一个Self-Boosting框架，用少量种子数据SFT模型自身作为提示生成器，在每一轮迭代中，提示生成器根据随机关键词产出合成prompt，用上一轮模型生成拒绝回答，再由响应改进器（同一模型经 SFT 区别学习 seed-answer 与当前输出的差异）改写成偏好回答，构成合成偏好对，在合成数据上重新训练模型，重复迭代。

- [**Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge**](https://arxiv.org/abs/2407.19594) *Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, Sainbayar Sukhbaatar.* Arxiv 2024.<br>让LLM扮演三个角色：生成回答的actor、为回答打分的judge和评估judge打分的meta-judge。模型通过自我打分和评估来生成偏好数据，自我迭代训练过程，不断生成新的训练数据并进行优化。

## 其他
****
- [**SELF-GUIDE: Better Task-Specific Instruction Following via Self-Synthetic Finetuning**](https://arxiv.org/abs/2407.12874) *Chenyang Zhao, Xueying Jia, Vijay Viswanathan, Tongshuang Wu, Graham Neubig.* COLM 2024.<br>让LLM用极少人类示例自合成大量任务专属训练数据，通过温度调节、噪声/长度规则过滤两轮质检，筛掉低质量样本，再用这些数据微调自身，从而无需外部标注或更强模型即可显著提升特定任务的指令遵循能力。

# Agent and Tool Use
核心是生成高质量的训练数据，教会大型语言模型如何像一个智能体（Agent）一样，通过调用外部工具（如APIs、代码解释器、数据库）来完成超越其自身固有能力的复杂任务。
## 构建工具/API使用的指令数据集
- [**Toolformer: Language Models Can Teach Themselves to Use Tools**](https://arxiv.org/abs/2302.04761) *Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom.* NeurIPS 2023.<br>提出一种让LLM自学习使用工具的方法，通过在文本中采样潜在的API调用位置，执行并验证结果后，将成功的调用作为新样本来微调模型自身。

- [**Gorilla: Large Language Model Connected with Massive APIs**](https://arxiv.org/abs/2305.15334) *Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez.*  NeurIPS 2024.<br>通过API文档使用LLM构建API指令数据集，并引入检索器感知的训练方法，让Gorilla能够在API文档更新时保持其输出的准确性和相关性，显著提升了模型调用API的准确性。

- [**GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction**](https://arxiv.org/abs/2305.18752) *Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, Ying Shan.* NeurIPS 2025.<br>提出一种自动化生成工具使用指令数据的方法，通过向ChatGPT提供多模态上下文（包括图像内容和工具描述）来生成工具相关的指令数据集。这些指令数据集包含了如何使用各种工具的指导。

- [**ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**](https://arxiv.org/abs/2307.16789) *Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, Maosong Sun.* ICLR 2024.<br>从RapidAPI Hub收集了大量真实世界的APIs，并让ChatGPT生成涉及这些APIs的多样化指令，包括单工具和多工具场景以及解决方案路径，从而构建了一个迄今为止最大、最多样的工具使用数据集ToolBench，涵盖了16000多个真实世界的API，并提出了一种深度优先搜索的决策树方法来评估工具调用的多步骤执行路径。

- [**ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases**](https://arxiv.org/abs/2306.05301) *Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, Le Sun.* Arxiv 2023.<br>从互联网上收集潜在有价值的工具的简要介绍，并让LLM生成文档，通过多智能体交互（用户代理、助手代理和工具执行代理）构建了一个包含工具使用场景的指令微调数据集，以增强模型的泛化工具学习能力。

## 生成复杂智能体流程的数据
这类工作超越了简单的单步工具调用，专注于生成包含多步推理、规划、执行和自我修正的复杂工作流（Agentic Flows）数据。

- [**ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving**](https://arxiv.org/abs/2309.17452) *Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, Weizhu Chen.* ICLR 2024.<br>训练了一个能同时生成解题步骤和调用计算工具（如代码解释器）的数学推理智能体，通过输出-反馈循环的训练方式，使其学会在解题过程中无缝地结合自然语言推理和精确计算。

- [**AgentInstruct: Toward Generative Teaching with Agentic Flows**](https://arxiv.org/abs/2407.03502) *Arindam Mitra, Luciano Del Corro, Guoqing Zheng, Shweti Mahajan, Dany Rouhana, Andres Codas, Yadong Lu, Wei-ge Chen, Olga Vrousgos, Corby Rosset, Fillipe Silva, Hamed Khanpour, Yash Lara, Ahmed Awadallah.* Arxiv 2024.<br>提出了一种新的数据生成范式，即生成智能体流程（Agentic Flows）而非简单的问答对，这些流程数据包含了任务分解、多工具协同、结果验证和自我修正等一系列步骤。通过让一个教师LLM生成智能体流程数据来完成一个复杂任务，并将这个完整的执行轨迹作为高质量的指令数据。

- [**AgentTuning: Enabling Generalized Agent Abilities for LLMs**](https://arxiv.org/abs/2310.12823) *Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, Jie Tang.* Findings of ACL 2024.<br>该工作认为Agent能力也应通过指令微调来获得。构建了一个轻量但高质量的、涵盖6个不同Agent任务的数据集AgentInstruct，通过设计统一的agent训练集和负样本策略进行能力分解调优，增强模型在新环境中的泛化能力。



## 与环境交互的具身智能
这类工作将LLM作为大脑，驱动一个智能体在有状态的环境（如游戏、操作系统）中进行探索和学习。数据来源于智能体与环境的真实交互过程。

- [**Voyager: An Open-Ended Embodied Agent with Large Language Models**](https://arxiv.org/abs/2305.16291) *Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar.* TMLR 2024.<br>展示了首个由LLM驱动、能在Minecraft中进行终身开放式探索的具身智能体。Voyager不依赖任何预设的训练数据，而是通过与环境互动来自主生成任务、编写代码来执行任务，并将成功的代码存入可复用的技能库，从而持续不断地生成和积累自己的训练数据。

- [**Efficient Agent Training for Computer Use**](https://arxiv.org/abs/2505.13909) *Yanheng He, Jiahe Jin, Pengfei Liu.* Arxiv 2025.<br> 专注于如何高效训练能操作电脑图形用户界面（GUI）的智能体，通过模仿学习和课程学习等方法，以少量人工鼠标键盘轨迹为起点并用LLM扩充多样化操作决策，训练电脑使用代理高效完成任务。

- [**Executable Code Actions Elicit Better LLM Agents**](https://arxiv.org/abs/2402.01030) *Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, Heng Ji.* ICML 2024.<br>论证了相比于文本，使用可执行代码作为LLM智能体的动作空间能带来更好的性能，因为代码具有明确的语法和执行反馈，可以减少幻觉并简化动作空间。

## 数据生成框架与工作流
这类工作不直接提供数据集，而是提供一个元工具或框架，让研究人员和开发者可以更轻松、更可复现地构建自己的合成数据生成流水线。

- [**Distilabel: An AI Feedback (AIF) Framework for Building Datasets with and for LLMs**](https://github.com/argilla-io/distilabel) *Álvaro Bartolomé Del Canto, Gabriel Martín Blázquez, Agustín Piqueres Lajarín and Daniel Vila Suero.* GitHub 2024.<br>是一个专注于AI反馈（AIF）的开源框架，旨在取代昂贵的人类反馈。它提供了构建数据集的完整流水线，让开发者可以使用强大的教师LLM（如GPT-4）来为学生LLM的输出打分、排序或提供反馈，从而高效地生成用于对齐任务的偏好数据集。

- [**Fuxion: Synthetic Data Generation and Normalization Functions using Langchain + LLMs**](https://github.com/tobiadefami/fuxion)<br> 一个利用Langchain和LLM来简化合成数据生成与规范化流程的Python库，专注于将非结构化数据转化为结构化输出。

- [**DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows**](https://arxiv.org/abs/2402.10379) *Ajay Patel, Colin Raffel, Chris Callison-Burch.* ACL 2024.<br>提供了一个开源Python库，旨在简化和标准化合成数据的生成过程。它允许用户通过模块化的方式组合各种LLM调用、提示策略和后处理步骤，从而快速创建、迭代和分享可复现的数据生成工作流。

