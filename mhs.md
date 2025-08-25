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

*****

- [**Orca 2: Teaching Small Language Models How to Reason**](https://arxiv.org/abs/2311.11045) *Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, Ahmed Awadallah.* Arxiv 2023.<br>教师模型被精心设计的系统提示所引导，为各种复杂任务生成详尽的、分步的解释和解决方案。这些富含逻辑、推理和上下文信息的解释轨迹构成了高质量的合成训练数据集。学生模型随后在这个数据集上进行微调，其目标不仅仅是模仿教师的最终答案，更是学习其解决问题的思维过程。

## 指令演化
这类方法的核心在于对指令本身进行优化和创造。它们从一个初始的、可能很简单的指令集出发，通过类似生物进化的变异和选择过程，自动地生成更复杂、更多样、更高质量的新指令，从而构建出强大的指令微调数据集。

- [**WizardLM: Empowering Large Language Models to Follow Complex Instructions**](https://arxiv.org/abs/2304.12244) *Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, Daxin Jiang.* Arxiv 2023.<br>提出了Evol-Instruct指令演化的方法，通过人工撰写的策略提示词，将简单的初始指令逐步演化出更复杂和多样化的指令数据。该方法结合了深入演化（增加指令复杂性）和广度演化（生成新的指令类型），并通过淘汰演化机制过滤掉低质量的指令，最终生成高质量的指令数据用于模型微调。

- [**Automatic Instruction Evolving for Large Language Models**](https://arxiv.org/abs/2406.00770) *Weihao Zeng, Can Xu, Yingxiu Zhao, Jian-Guang Lou, Weizhu Chen.* EMNLP 2024.<br>借鉴进化计算的思想，自动地将简单、低质量的指令进化成更复杂、高质量的指令。通过多轮的指令生成、变异、筛选和模型再训练，可以自动地创造出一个多样且高质量的指令数据集。
*****
- [**Tag-Evol: Achieving Efficient Instruction Evolving via Tag Injection**](https://arxiv.org/pdf/2406.00770) *Weihao Zeng, Can Xu, Yingxiu Zhao, Jian-Guang Lou, Weizhu Chen.* Findings of ACL 2025.<br>提出了 Tag-Evol 框架，用 LLM 先对种子数据打上细粒度知识标签，再在重写指令时按预算把若干标签注入，通过知识标签信息注入来实现高效且多样化的指令演化。

## 多智能体
这类方法利用多个智能体之间的交互来生成数据。

- [**CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society**](https://arxiv.org/abs/2303.17760) *Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, Bernard Ghanem.* NeurIPS 2023.<br>提出了一个新颖的沟通式智能体框架，让一个AI用户和一个AI助手通过角色扮演进行对话来完成任务，从而大规模地生成能反映合作行为和指令遵循能力的对话数据。

*****
- [**Bootstrapping LLM-based Task-Oriented Dialogue Agents via Self-Talk**](https://aclanthology.org/2024.findings-acl.566.pdf) *Dennis Ulmer, Elman Mansimov, Kaixiang Lin, Justin Sun, Xibin Gao, Yi Zhang.* Findings of ACL 2024.<br>通过两个LLM分别代表客户端和代理模型，通过给客户端和代理模型提供角色描述、行为指令和对话历史，让它们在指定的角色中进行对话，并引入过滤步骤，保留质量较高的对话作为训练数据。

- [**Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation**](https://aclanthology.org/2025.acl-long.1136.pdf) *Shuo Tang, Xianghe Pang, Zexi Liu, Bohan Tang, Rui Ye, Tian Jin, Xiaowen Dong, Yanfeng Wang, Siheng Chen.* ACL 2025.<br>提出了MATRIX，一个多智能体模拟器，其使用基于真实人类配置文件的智能体，赋予它们目标和生活目标，来模拟人类生活场景，并基于MATRIX生成的场景来生成高度真实和可控的合成指令数据。

- [**A Strategic Coordination Framework of Small LLMs Matches Large LLMs in Data Synthesis**](https://aclanthology.org/2025.acl-long.566/) *Xin Gao, Qizhi Pei, Zinan Tang, Yu Li, Honglin Lin, Jiang Wu, Lijun Wu, Conghui He.* ACL 2025.<br>提出一个名为GRA的协作框架来解决如何利用多个小型语言模型协作达到与大型语言模型相当的数据质量的问题。GRA框架通过协调三个专门的角色来运作：生成器、评审者和仲裁者。生成器负责产生初始样本，评审者对样本的质量进行评估，仲裁者则解决评审者之间的冲突并最终确定输出。此外，GRA还包含一个后处理模块，用于通过嵌入去重和元数据丰富来进一步优化结果。

## 自生成奖励与偏好
这类方法让LLM自己充当裁判或奖励模型，来为生成的内容打分或提供偏好判断。

- [**Constitutional AI: Harmlessness from AI Feedback**](https://arxiv.org/abs/2212.08073) *Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, et al.* Arxiv 2022.<br>提出Constitutional AI对齐策略，通过人工预先制定一组原则让模型依据这些AI宪法自我监督，从而无需人工逐条标注有害内容。分为两个阶段，在监督微调阶段，让模型从初始模型输出中生成自我批判和修正的响应并据此微调模型；随后在强化学习阶段，让模型自身比较两种输出优劣生成偏好数据来训练奖励模型，并使用该奖励信号进行策略优化。

- [**Self-Rewarding Language Models.**](https://arxiv.org/abs/2401.10020) *Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston.* ICML 2024.<br>提出一个训练框架，其中LLM在作为奖励模型为自己生成的多个响应进行打分后，再利用这些自我生成的奖励通过DPO等算法进行微调，实现了自给自足的对齐学习。

****
- [**Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge**](https://arxiv.org/abs/2407.19594) *Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, Sainbayar Sukhbaatar.* Arxiv 2024.<br>让LLM扮演三个角色：生成回答的actor、为回答打分的judge和评估judge打分的meta-judge。模型通过自我打分和评估来生成偏好数据，自我迭代训练过程，不断生成新的训练数据并进行优化。

- [**Spread Preference Annotation: Direct Preference Judgment for Efficient LLM Alignment**](https://arxiv.org/abs/2406.04412v2) *Dongyoung Kim, Kimin Lee, Jinwoo Shin, Jaehyung Kim.* ICLR 2025.<br>用极少量人工偏好数据作为种子，通过迭自采样-自打分-自精炼三步，直接利用当前模型 logits 生成偏好标签并去噪，再用 DPO 微调，持续扩散人类偏好先验，实现低成本高效对齐。

- [**Self-Boosting Large Language Models with Synthetic Preference Data**](https://arxiv.org/abs/2410.06961) *Qingxiu Dong, Li Dong, Xingxing Zhang, Zhifang Sui, Furu Wei.* ICLR 2025.<br>提出了一个Self-Boosting框架，用少量种子数据SFT模型自身作为提示生成器，在每一轮迭代中，提示生成器根据随机关键词产出合成prompt，用上一轮模型生成拒绝回答，再由响应改进器（同一模型经 SFT 区别学习 seed-answer 与当前输出的差异）改写成偏好回答，构成合成偏好对，在合成数据上重新训练模型，重复迭代。

- [**An Uncertainty-Driven Adaptive Self-Alignment Framework for Large Language Models**](https://arxiv.org/abs/2507.17477) *Haoran Sun, Zekun Zhang, Shaoning Zeng.* Arxiv 2025.<br>提出了UDASA框架，其通过为每个输入生成多个响应，并从语义、事实和价值观对齐三个维度量化这些响应的不确定性，实现了自我对齐。基于不确定性差异，UDASA 自动构建偏好对，并将训练样本分为保守、中等和探索三个阶段，逐步优化模型。

## 其他
****

- [**Self-instruct: Aligning language models with self-generated instructions**](https://arxiv.org/abs/2212.10560) *Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi.* ACL 2023.<br>从一个小型的人类种子任务集开始，迭代生成新的任务指令和对应的输入输出实例，然后通过过滤低质量或重复的生成内容，最终使用生成的数据对原始模型进行微调，从而显著提高模型在遵循指令方面的性能。

- [**SELF-GUIDE: Better Task-Specific Instruction Following via Self-Synthetic Finetuning**](https://arxiv.org/abs/2407.12874) *Chenyang Zhao, Xueying Jia, Vijay Viswanathan, Tongshuang Wu, Graham Neubig.* COLM 2024.<br>让LLM用极少人类示例自合成大量任务专属训练数据，通过温度调节、噪声/长度规则过滤两轮质检，筛掉低质量样本，再用这些数据微调自身，从而无需外部标注或更强模型即可显著提升特定任务的指令遵循能力。

- [**CodecLM: Aligning Language Models with Tailored Synthetic Data**](https://aclanthology.org/2024.findings-naacl.235.pdf) *Zifeng Wang, Chun-Liang Li, Vincent Perot, Long T. Le, Jin Miao, Zizhao Zhang, Chen-Yu Lee, Tomas Pfister.* Findings of NAACL 2024.<br>CodecLM通过利用LLM作为编解码器，将种子指令编码为元数据，再解码生成定制化指令。它引入Self-Rubrics根据元数据生成评估标准和改进动作，提升指令复杂性；同时采用Contrastive Filtering筛选最有效的指令-响应对，优化数据质量，从而提高LLM在特定任务上的指令跟随能力。

- [**CoT-Self-Instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks**](https://arxiv.org/abs/2507.23751) *Ping Yu, Jack Lanchantin, Tianlu Wang, Weizhe Yuan, Olga Golovneva, Ilia Kulikov, Sainbayar Sukhbaatar, Jason Weston, Jing Xu.* Arxiv 2025.<br>让LLMs基于给定的种子任务进行思维链推理和规划），生成新的合成指令。随后，利用自动化的筛选方法（如 Answer-Consistency 和 Rejecting Instruction Preferences, RIP）对生成的数据进行筛选，以确保数据质量。

- [**DataGen: Unified Synthetic Dataset Generation via Large Language Models**](https://arxiv.org/abs/2406.18966) *Yue Huang, Siyuan Wu, Chujie Gao, Dongping Chen, Qihui Zhang, Yao Wan, Tianyi Zhou, Xiangliang Zhang, Jianfeng Gao, Chaowei Xiao, Lichao Sun.* ICLR 2025.<br>提出DataGen，一个利用LLM统一生成多种类型高质量数据集的框架。DataGen通过创新机制提升生成数据的多样性和准确性：利用属性引导和组校验确保生成数据具备丰富多样的风格；采用代码执行来验证标签准确、结合检索增强保证事实正确；并允许用户指定约束以定制生成过程。

# Agent and Tool Use
核心是生成高质量的训练数据，教会大型语言模型如何像一个智能体（Agent）一样，通过调用外部工具（如APIs、代码解释器、数据库）来完成超越其自身固有能力的复杂任务。
## Evaluation Dataset
- [**API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs**](https://arxiv.org/abs/2304.08244) *Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, Yongbin Li.* EMNLP 2023.<br>论文创建了一个名为API-Bank的基准测试，它专门设计用于评估工具增强型LLM，API-Bank包括一个可运行的评估系统，由73个API工具组成，并对314个工具使用对话进行了注释，包含753个API调用，以评估现有LLMs在规划、检索和调用APIs方面的能力。

- [**WebArena: A Realistic Web Environment for Building Autonomous Agents**](https://arxiv.org/abs/2307.13854) *Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, Graham Neubig.* NeurIPS 2023.<br> 提出了WebArena，一个为开发和测试自主智能代理而设计的高度真实且可复现的网络环境。它包含四个常见领域的完全功能网站，以及工具和外部知识库，以支持类似人类的任务解决过程。论文还发布了一套包含812个测试示例的基准测试任务，专注于评估任务完成的功能性正确性。

- [**ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**](https://arxiv.org/abs/2307.16789) *Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, Maosong Sun.* ICLR 2024.<br>从RapidAPI Hub收集了大量真实世界的APIs，并让ChatGPT生成涉及这些APIs的多样化指令，包括单工具和多工具场景以及解决方案路径，从而构建了一个迄今为止最大、最多样的工具使用数据集ToolBench，涵盖了16000多个真实世界的API，并提出了一种深度优先搜索的决策树方法来评估工具调用的多步骤执行路径。

- [**StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models**](https://aclanthology.org/2024.findings-acl.664.pdf) *Zhicheng Guo, Sijie Cheng, Hao Wang, Shihao Liang, Yujia Qin, Peng Li, Zhiyuan Liu, Maosong Sun, Yang Liu.* Findings of ACL 2024.<br>StableToolBench的主要目的是评估LLMs在工具学习任务中的性能，特别是模型如何使用外部工具来解决复杂问题。该数据集通过虚拟API服务器和稳定的评估系统，确保评估过程的稳定性和结果的可重复性，同时保持任务的现实感，并引入了新的评估指标（如 Solvable Pass Rate 和 Solvable Win Rate）来专门用于评估模型在可解任务上的表现。

- [**AgentBench: Evaluating LLMs as Agents**](https://arxiv.org/abs/2308.03688) *Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, Jie Tang.* ICLR 2024.<br>提出了AGENTBENCH，这是一个多维度的、不断发展的基准测试，来评估LLMs作为智能代理在交互环境中处理复杂任务的能力。研究者们构建了一个包含8个不同环境的基准测试，涵盖代码、游戏和网络交互等多种场景，用以评估LLMs在执行指令、编码、知识获取、逻辑推理和常识理解等方面的能力。

- [**GAIA: a benchmark for General AI Assistants**](https://arxiv.org/abs/2311.12983) *Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, Thomas Scialom.* ICLR 2024.<br>GAIA通过设计一系列真实世界的问题，要求AI助手具备推理、多模态处理、网页浏览和工具使用等基本能力，这些问题对人类来说概念上简单，但对大多数先进的AI系统来说具有挑战性。问题根据解决所需的步骤数量和工具种类分为三个难度级别，从简单的Level 1到复杂的Level 3，并通过模型给出的答案与ground truth对比进行自动化评估。

- [**MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use**](https://arxiv.org/abs/2310.03128) *Yue Huang, Jiawen Shi, Yuan Li, Chenrui Fan, Siyuan Wu, Qihui Zhang, Yixin Liu, Pan Zhou, Yao Wan, Neil Zhenqiang Gong, Lichao Sun.* ICLR 2024.<br>提出了一个名为METATOOL的基准测试，用来评估LLMs是否具有工具使用意识以及能否正确选择工具，其创建了一个包含21,127个用户查询的TOOLE数据集，涵盖单工具和多工具场景，并设计了四个子任务来从不同角度评估工具选择能力。


- [**ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities**](https://aclanthology.org/2025.findings-naacl.65.pdf) *Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer
Feng Nan, Felix Bai, Shuang Ma, Shen Ma, Mengyu Li,
Guoli Yin, Zirui Wang, Ruoming Pang.* Findings of NAACL 2025.<br>提出了一个名为TOOLSANDBOX的评估基准，旨在全面评估LLMs在使用工具来解决现实世界挑战时的能力。它通过引入状态化工具执行、对话性评估和互动性评估，克服了现有基准测试的局限性。ToolSandbox包含1032个精心设计的测试案例，涵盖多种复杂场景，如状态依赖、规范化和信息不足等，揭示了即使是性能最好的模型也面临挑战。

- [**DABstep: Data Agent Benchmark for Multi-step Reasoning**](https://arxiv.org/abs/2506.23719) *Alex Egg, Martin Iglesias Goyanes, Friso Kingma, Andreu Mora, Leandro von Werra, Thomas Wolf.* Arxiv 2025.<br>提出了DABstep，这是一个用于评估agent在现实多步数据分析任务上的新基准测试。DABstep包含超过450个真实世界的数据分析任务，这些任务直接来源于Adyen的金融分析平台，其要求agent结合基于代码的数据处理和对异构文档的上下文推理，测试数据操作、跨多个来源交叉引用和精确结果报告的能力。DABstep提供了事实性答案格式和自动正确性检查，以实现大规模的客观评分。

- [**FamilyTool: A Multi-hop Personalized Tool Use Benchmark**](https://arxiv.org/abs/2504.06766) *Yuxin Wang, Yiran Guo, Yining Zheng, Zhangyue Yin, Shuo Chen, Jie Yang, Jiajun Chen, Yuan Li, Xuanjing Huang, Xipeng Qiu.* Arxiv 2025.<br>提出了一个名为FamilyTool的多跳个性化工具使用基准测试，其基于一个家庭知识图谱，模拟了需要多跳推理和归纳知识适应的动态环境中的个性化工具使用场景。此外，文章还提出了KGETool，这是一个简单的知识图谱增强的评估流程，用于系统地评估LLMs在这些场景中的工具使用能力。

- [**T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning**](https://arxiv.org/abs/2505.16986) *Amartya Chakraborty, Paresh Dashore, Nadia Bathaee, Anmol Jain, Anirban Das, Shi-Xiong Zhang, Sambit Sahu, Milind Naphade, Genta Indra Winata.* Arxiv 2025.<br>介绍了T1，一个用于评估多轮对话中工具使用和规划能力的多领域对话数据集。它包含13.5k对话，涵盖九个领域（包括单领域和多领域设置），并引入缓存机制以支持工具调用结果的重用。

- [**The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models**](https://openreview.net/pdf?id=2GmDdhBdDk) *Shishir G Patil
, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, Joseph E. Gonzalez.* ICML 2025.<br>介绍了BFCL，这是一个用于评估LLM在多种真实场景下调用外部函数能力的综合基准测试。BFCL通过结合专家策划和用户贡献的函数及提示，评估模型在串行和并行函数调用、多种编程语言以及多步代理设置中的能力，使用抽象语法树评估方法实现可扩展性。

- [**The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs**](https://aclanthology.org/2025.findings-acl.1205.pdf) *Avinash Baidya, Kamalika Das, Xiang Gao.* Findings of ACL 2025.<br>提出了一种评估框架，用于量化LLM智能体与人类专家在复杂任务导向型对话系统中的行为差距。该框架通过对话行为、工具使用和知识利用三个维度进行评估，并采用教师强制方法进行控制评估，以避免用户模拟器引入的额外差异。

## Synthetic Dataset
### 构建工具/API使用的指令数据集
- [**ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases**](https://arxiv.org/abs/2306.05301) *Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, Le Sun.* Arxiv 2023.<br>从互联网上收集潜在有价值的工具的简要介绍，并让LLM生成文档，通过多智能体交互（用户代理、助手代理和工具执行代理）构建了一个包含工具使用场景的指令微调数据集，以增强模型的泛化工具学习能力。

- [**Toolformer: Language Models Can Teach Themselves to Use Tools**](https://arxiv.org/abs/2302.04761) *Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom.* NeurIPS 2023.<br>提出一种让LLM自学习使用工具的方法，通过在文本中采样潜在的API调用位置，执行并验证结果后，将成功的调用作为新样本来微调模型自身。

- [**Gorilla: Large Language Model Connected with Massive APIs**](https://arxiv.org/abs/2305.15334) *Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez.*  NeurIPS 2024.<br>通过API文档使用LLM构建API指令数据集，并引入检索器感知的训练方法，让Gorilla能够在API文档更新时保持其输出的准确性和相关性，显著提升了模型调用API的准确性。

- [**APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets"**](https://arxiv.org/abs/2406.18518) *Zuxin Liu, Thai Hoang, Jianguo Zhang, Ming Zhu, Tian Lan, Shirley Kokane, Juntao Tan, Weiran Yao, Zhiwei Liu, Yihao Feng, Rithesh Murthy, Liangwei Yang, Silvio Savarese, Juan Carlos Niebles, Huan Wang, Shelby Heinecke, Caiming Xiong.* Arxiv 2024.<br>通过从API库中采样API和示例问答对，并利用LLM根据多样化的提示模板生成相关的问答对，从而创建合成数据集。生成的数据经过严格的多阶段验证，包括格式检查、实际函数执行和语义验证，以确保数据的质量和可靠性。

- [**GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction**](https://arxiv.org/abs/2305.18752) *Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, Ying Shan.* NeurIPS 2023.<br>提出一种自动化生成工具使用指令数据的方法，通过向ChatGPT提供多模态上下文（包括图像内容和工具描述）来生成工具相关的指令数据集。这些指令数据集包含了如何使用各种工具的指导。

- [**ToolGrad: Efficient Tool-use Dataset Generation with Textual "Gradients"**](https://arxiv.org/abs/2508.04086) *Zhongyi Zhou, Kohei Uehara, Haoyu Zhang, Jingtao Zhou, Lin Gu, Ruofei Du, Zheng Xu, Tatsuya Harada.* Arxiv 2025.<br>论文提出了一个名为 ToolGrad 的框架，核心思想是逆转传统方法的流程：先生成有效的工具使用链，再合成对应的用户查询，而不是先生成用户查询再寻找工具使用解决方案。

- [**Enhancing LLM Tool Use with High-quality Instruction Data from Knowledge Graph**](https://arxiv.org/abs/2506.21071) *Jingwei Wang, Zai Zhang, Hao Qian, Chunjing Gan, Binbin Hu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou, Bin Shi, Bo Dong.* Arxiv 2025.<br>提出了一种利用知识图谱生成高质量指令数据的方法，以增强LLM的工具使用能力。该方法通过将知识图谱中的关系类型抽象为API，并使用一阶逻辑（FOL）查询从知识图谱中采样复杂的子图，生成自然语言查询和相应的解决方案路径。这些查询和解决方案路径被转换为对话格式，形成指令数据集KG2Tool。
### 多轮agent对话合成

- [**APIGen-MT: Agentic PIpeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay**](https://arxiv.org/abs/2504.03601) *Akshara Prabhakar, Zuxin Liu, Ming Zhu, Jianguo Zhang, Tulika Awalgaonkar, Shiyu Wang, Zhiwei Liu, Haolin Chen, Thai Hoang, Juan Carlos Niebles, Shelby Heinecke, Weiran Yao, Huan Wang, Silvio Savarese, Caiming Xiong.* Arxiv 2025.<br>提出了一种两阶段的数据合成方法来生成高质量的多轮人机交互数据。在第一阶段，通过上下文准备、基于LLM的数据生成器、格式和执行检查、评审委员会以及反馈生成和改进等步骤，生成详细的任务蓝图，包括用户意图、可验证的地面真实动作和预期的最终输出。在第二阶段，基于第一阶段生成的蓝图，通过模拟人机交互来生成完整的多轮交互轨迹，包括对话轮次、代理动作和环境响应，并通过验证确保轨迹的正确性和合理性。

- [**Magnet: Multi-turn tool-use data synthesis and distillation via graph translation**](https://aclanthology.org/2025.acl-long.1566.pdf) *Fan Yin, Zifeng Wang, I-Hung Hsu, Jun Yan, Ke Jiang, Yanfei Chen, Jindong Gu, Long T. Le, Kai-Wei Chang, Chen-Yu Lee, Hamid Palangi, Tomas Pfister.* ACL 2025.<br>提出了名为Magnet的数据合成方法，其通过构建局部依赖图来组织函数间的依赖关系，并利用随机游走生成初始函数签名路径。在此基础上，运用插入、合并和分割三种节点操作来增强路径，以覆盖多轮交互中的复杂场景。最后，通过教师模型生成正向轨迹，并构造负向轨迹，为模型训练提供高质量的对比学习样本。

### 生成复杂智能体流程的数据
这类工作超越了简单的单步工具调用，专注于生成包含多步推理、规划、执行和自我修正的复杂工作流（Agentic Flows）数据。

- [**ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving**](https://arxiv.org/abs/2309.17452) *Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, Weizhu Chen.* ICLR 2024.<br>提出了TORA，通过无缝结合自然语言推理和外部工具的使用来解决复杂数学问题。其通过设计交互式工具使用轨迹，利用模仿学习训练模型以生成高质量的推理轨迹，以及通过输出空间塑形技术进一步优化模型的推理行为，从而提升模型在数学问题求解中的性能和灵活性。为了生成这些交互式工具使用轨迹。论文利用GPT-4合成了高质量的标注数据，这些数据不仅包含自然语言推理步骤，还包含程序代码和工具执行的输出，为模型训练提供了丰富的样本。

- [**AgentInstruct: Toward Generative Teaching with Agentic Flows**](https://arxiv.org/abs/2407.03502) *Arindam Mitra, Luciano Del Corro, Guoqing Zheng, Shweti Mahajan, Dany Rouhana, Andres Codas, Yadong Lu, Wei-ge Chen, Olga Vrousgos, Corby Rosset, Fillipe Silva, Hamed Khanpour, Yash Lara, Ahmed Awadallah.* Arxiv 2024.<br>提出了一种新的数据生成范式，即生成智能体流程（Agentic Flows）而非简单的问答对，这些流程数据包含了任务分解、多工具协同、结果验证和自我修正等一系列步骤。其提出了一种生成性教学（Generative Teaching）的思想，通过使用原始数据源（如文本文档和代码文件）作为种子，自动创建大量多样化且高质量的合成数据。这些数据通过内容转换流程、种子指令生成流程和指令细化流程生成，覆盖多种技能，用于教授模型新技能或行为，从而提高模型性能。

- [**AgentTuning: Enabling Generalized Agent Abilities for LLMs**](https://arxiv.org/abs/2310.12823) *Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, Jie Tang.* Findings of ACL 2024.<br>通过任务衍生和自我指令的方法构建了AgentInstruct数据集，涵盖了六个不同代理任务的高质量交互轨迹。这些轨迹通过GPT-4与环境的交互生成，并根据奖励分数进行过滤以确保数据质量。最后，结合一般领域的开源指令，通过混合指令调整策略对LLMs进行微调，以提升其代理能力并保持通用性能。

****
- [**Re-ReST: Reflection-Reinforced Self-Training for Language Agents**](https://aclanthology.org/2024.emnlp-main.861.pdf) *Zi-Yi Dou, Cheng-Fu Yang, Xueqing Wu, Kai-Wei Chang, Nanyun Peng.* EMNLP 2024.<br>通过引入反思模型把自我训练中agent自我生成的低质量样本，利用环境反馈（如单元测试结果）修正为高质量样本，从而低成本地扩充训练集。

- [**ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent**](https://arxiv.org/abs/2312.10003) *Renat Aksitov, Sobhan Miryoosefi, Zonglin Li, Daliang Li, Sheila Babayan, Kavya Kopparapu, Zachary Fisher, Ruiqi Guo, Sushant Prakash, Pranesh Srinivasan, Manzil Zaheer, Felix Yu, Sanjiv Kumar.* ICLR 2024.<br>论文将 ReAct 推理流程拆解为多个可微调的步骤轨迹，并在固定问题集上循环执行 ReST 式自我改进，通过轨迹生成，并用 LLM as a judge 反馈进行持续的自我改进和自我蒸馏。

### 与环境交互的智能体
这类工作将LLM作为大脑，驱动一个智能体在有状态的环境中进行探索和学习。数据来源于智能体与环境的真实交互过程。

- [**Executable Code Actions Elicit Better LLM Agents**](https://arxiv.org/abs/2402.01030) *Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, Heng Ji.* ICML 2024.<br>通过筛选和改造现有数据集（如HotpotQA、APPS、MATH、WikiTableQuestions和ALFWorld），生成了包含7k多轮交互轨迹的CodeActInstruct数据集。该方法将单轮问题转化为多轮交互问题，并选择性保留那些模型最初出错但后续修正的轨迹，以增强LLM代理的自我改进能力。

- [**Efficient Agent Training for Computer Use**](https://arxiv.org/abs/2505.13909) *Yanheng He, Jiahe Jin, Pengfei Liu.* Arxiv 2025.<br> 专注于如何高效训练能操作电脑图形用户界面（GUI）的智能体，通过模仿学习和课程学习等方法，以少量人工鼠标键盘轨迹为起点，利用先进的推理模型（如Claude 3.7 Sonnet）在每个真实人类轨迹步骤的基础上，生成多样化的行动决策及其对应的思考过程，从而训练电脑使用代理高效完成任务。

### 数据生成框架与工作流
这类工作不直接提供数据集，而是提供一个元工具或框架，让研究人员和开发者可以更轻松、更可复现地构建自己的合成数据生成流水线。

- [**Distilabel: An AI Feedback (AIF) Framework for Building Datasets with and for LLMs**](https://github.com/argilla-io/distilabel) *Álvaro Bartolomé Del Canto, Gabriel Martín Blázquez, Agustín Piqueres Lajarín and Daniel Vila Suero.* GitHub 2024.<br>是一个专注于AI反馈（AIF）的开源框架，旨在取代昂贵的人类反馈。它提供了构建数据集的完整流水线，让开发者可以使用强大的教师LLM（如GPT-4）来为学生LLM的输出打分、排序或提供反馈，从而高效地生成用于对齐任务的偏好数据集。

- [**Fuxion: Synthetic Data Generation and Normalization Functions using Langchain + LLMs**](https://github.com/tobiadefami/fuxion)<br> 一个利用Langchain和LLM来简化合成数据生成与规范化流程的Python库，专注于将非结构化数据转化为结构化输出。

- [**DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows**](https://arxiv.org/abs/2402.10379) *Ajay Patel, Colin Raffel, Chris Callison-Burch.* ACL 2024.<br>提供了一个开源Python库，旨在简化和标准化合成数据的生成过程。它允许用户通过模块化的方式组合各种LLM调用、提示策略和后处理步骤，从而快速创建、迭代和分享可复现的数据生成工作流。

