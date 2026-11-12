# LLM 数据生命周期研究调研 · 论文 Idea 一句话

> 调研日期：2026-11-11 · 语料：`data/papers.json` 中的 **2,807 篇** 去重论文（2026-08 ~ 2026-11，来自 11 次文献 bot 发现分支的合并去重）
> 方法：全量 2,807 篇的一句话 idea 由 `scripts/extract_ideas.py` 从 arXiv 摘要的贡献句确定性抽取，见 [`PAPERS.md`](PAPERS.md) 每条下的 `_Idea:_`；本文件的中文一句话为**逐篇精读**标题与摘要后撰写，覆盖相关性分数 ≥18 的核心 60 篇。

---

## 一、系统性调研归纳

### 1. 语料格局

| 维度 | 分布 |
| --- | --- |
| 总量 | 2,807 篇，覆盖 4 个月（8 月 606 / 9 月 885 / 10 月 1,139 / 11 月 177，11 月未满月） |
| 一级分层 | 数据摄入策略 **1,518 (54%)** > 数据基底 **1,107 (39%)** ≫ 数据创建与选择 **182 (6.5%)** |
| 最大子类 | 运行时检索/搜索/验证 1,288；未归类 Other 876；课程/中期训练/回放 187；智能体与工具使用 108；合成数据 93 |
| 相关性最高子类 | 数据选择 12.5、合成数据 12.2、数据混合 10.6（全体均值 8.9） |

月度上三个分层的占比相当稳定（摄入约 52–57%，基底约 37–42%，创建约 5–7%），说明这不是某个月的偶然偏斜，而是这套检索词族与分类体系下的持续结构性倾斜：社区 2026 下半年的产出重心明显在**「运行时如何用数据」**，而非「如何造数据」。

### 2. 标题词频反映的热点

| 主题词 | 出现次数 | 读法 |
| --- | --- | --- |
| agents / agentic / agent | 506 / 376 / 211 | 智能体是绝对主线，合计逾千次 |
| learning / reasoning | 229 / 227 | 推理与学习仍是与数据并列的核心动词 |
| benchmark / benchmarking / evaluation / evaluating | 206 / 153 / 114 / 112 | 评测基础设施化，合计约 585 次 |
| retrieval / retrieval-augmented / search | 178 / 66 / 79 | 检索仍是运行时信息使用的主要入口 |
| memory / knowledge / evidence | 120 / 100 / 62 | 外部记忆与证据接地 |
| long-horizon / test-time / verification | 97 / 83 / 93 | 长时程与测试时验证成为独立议题 |
| reinforcement / optimization / training | 83 / 100 / 79 | 强化学习与优化渗透到数据环节 |

### 3. 四条主要研究脉络

1. **智能体数据工程化**：从「收集轨迹」转向「可验证环境 + 可分解奖励 + 自博弈课程」。典型如 SPADE（自适应合成环境自博弈）、Explorer-Zero（任务与细粒度奖励联合生成）、Terminal-Universe（把轨迹反转为可复用环境）、LURE（追逐—规避式零数据课程）。数据不再是一次性语料，而是可以反复查询、提供执行反馈的基础设施。
2. **合成数据的质量与效用评估**：这是相关性分数最集中的区域。FROST（真实数据梯度锚定在线效用）、TATC（训练感知的目标覆盖）、CIGA（消除表层形式偏置的梯度归因）、Group-Level Signals（必须看组级信号）共同指向一个判断：**合成数据的瓶颈已从「能不能生成」变成「该不该训练、会不会造成遗忘」**。When Synthetic Data Hurts 直接给出域内提升、域外退化的实证。
3. **测试时信息使用与验证**：HSRM（读隐状态而非重读文本做验证）、VGSC（生成与验证双不完美信号的平衡）、ReTAC（残差式分配测试时计算）、AWM（无环境副本的测试时扩展）说明「验证」本身正在被当作一个需要单独建模与压缩成本的数据环节。
4. **评测基准的可执行化**：benchmark 类工作大量自带生成框架、固定知识库与可验证环境（PowerBench、ATLAS-Bench、TURTLE、DataSense-Bench），而不是静态问答集。评测对象也从「答案对不对」扩展到「过程质量、长时程工具使用、约束满足」。

### 4. 对仓库策展的建议

- 正文 267 条人工策展条目若要从这批发现中补充，**优先看下面 60 篇高分论文**，尤其「合成数据效用评估」与「可验证环境/轨迹生成」两条线，它们与仓库的三层 taxonomy 契合度最高。
- `Other` 子类有 876 篇、均值 8.0，多为分类边界外的边缘命中，建议人工过滤后再考虑入库，不宜整批晋升。
- 相关性分数来自关键词规则，衡量的是**主题契合度而非论文质量**；下面的中文概括已逐篇依据摘要撰写，可作为二次筛选依据。

### 5. 方法局限

本调研基于标题与 arXiv 摘要，未研读全文与实验细节；全量英文 idea 是规则抽取的贡献句，个别会抽到方法细节句或背景句；中文精读覆盖 score ≥18 的 60 篇，其余 2,747 篇仅有英文自动抽取句。分层由 bot 的确定性 taxonomy 规则给出，存在跨层论文被归到单一主层的情况。

---

## 二、核心 60 篇 · 中文一句话 Idea

按相关性分数降序；括号内为分数与一级分层/子类。

1. **[MidTool: Mid-Training Data Synthesis for Agentic Tool Use](http://arxiv.org/abs/2608.20314v1)** `34 · 摄入/中期训练` — 用专门合成的中期训练数据塑造 LLM 的通用工具使用能力，把它从已被充分研究的数理推理能力中独立出来。
2. **[DataCanvas-EDU: An Agentic Framework for Instructor-Guided Synthetic Data Generation in Business Analytics Education](http://arxiv.org/abs/2609.19617v1)** `26 · 创建/合成数据` — 让教师以教学目标与约束引导智能体框架，生成可核验、可适配课程的商业分析教学数据集。
3. **[Test-Time Confidence Adaptation under RL-Induced Trajectory Shift](http://arxiv.org/abs/2610.35500v1)** `26 · 摄入/运行时验证` — 提出 StaR-TTA，用稳定性引导的无标签测试时自适应，修复智能体强化学习造成推理轨迹漂移后的置信度估计失真。
4. **[Improving Synthetic Data Generation for Argument Mining via Adversarial Reinforcement Learning](http://arxiv.org/abs/2610.07699v1)** `25 · 创建/合成数据` — 用对抗式强化学习生成论证挖掘数据，同时保证结构标注准确与样本多样。
5. **[SPADE: Self-Play in Adaptive Synthetic Executable Environments](http://arxiv.org/abs/2608.19197v2)** `25 · 基底/预训练数据` — 用自博弈让「环境设计者」从预训练语料持续生成随学习者能力演进的可执行任务环境。
6. **[Let Training Guide Selection: Online Synthetic Data Filtering via Real-Anchored Utility](http://arxiv.org/abs/2609.29988v1)** `23 · 创建/数据选择` — 以真实训练数据上的梯度反馈为锚，在线估计每条合成数据对当前学习者的实际效用并筛选。
7. **[Qwen-Planner-Agent: A Closed-Loop AI-for-AI Framework for Real-World Mobile Planner Agents](http://arxiv.org/abs/2609.29892v1)** `23 · 摄入/运行时检索` — 构建「AI 造 AI」闭环，让模型参与自身的数据构建与迭代，落地为真实场景的移动规划智能体。
8. **[Agentic World Modeling for Agent Test-time Scaling without Environment Replica](http://arxiv.org/abs/2610.40927v1)** `22 · 摄入/运行时验证` — 把下一观测预测改写成多轮证据获取任务，使测试时多轨迹采样无需复制不可回滚的共享环境。
9. **[Evolving Self-tuning Benchmarks for Long-Horizon Tool Use](http://arxiv.org/abs/2610.27307v2)** `22 · 创建/合成数据` — 用「提议者—求解者—反馈生成器」的无监督环境设计自动演化长时程工具使用任务。
10. **[PowerBench: A Benchmark for Agentic Retrieval and Reasoning in Power Systems](http://arxiv.org/abs/2609.34492v1)** `22 · 基底/Other` — 用共同依赖链生成互连的异构运行数据，构造可评测智能体检索与推理的电力系统基准。
11. **[SelectTTRL: Representation-Guided Data Selection for Efficient Test-Time Reinforcement Learning](http://arxiv.org/abs/2610.16104v1)** `22 · 摄入/运行时验证` — 在昂贵的测试时训练之前，用表示信号无标签识别哪些题目真正值得训练。
12. **[Agent-R²: Scaling Agents for Realistic Productivity Work via Real-Source Data Synthesis](http://arxiv.org/abs/2611.08219v1)** `21 · 创建/合成数据` — 从真实生产流程与异构文件合成训练数据，让智能体胜任长时程专业办公任务。
13. **[LentEx: Generalizable Latent Entity Extraction via Synthetic Data and Instruction-Tuned LLMs](http://arxiv.org/abs/2609.04511v1)** `21 · 创建/合成数据` — 用合成数据加指令微调训练小模型，识别文本中隐含、需上下文推断的实体。
14. **[SCOUT: Synergizing Reasoning and Tool-Use for Computer-Use Safety](http://arxiv.org/abs/2609.36201v1)** `21 · 摄入/运行时验证` — 两阶段安全验证器：先做推理密集的评分细则生成，再做工具密集的证据收集。
15. **[Scaling Agentic Data for Long-Horizon Terminal Intelligence](http://arxiv.org/abs/2610.13166v1)** `21 · 摄入/运行时检索` — 全自动数据合成引擎，产出长时程终端智能体所需的「探索—判断—调整—交付」完整轨迹。
16. **[What Makes Good Agentic Data? An ACE Lens on Data Generation for LLM Agents](http://arxiv.org/abs/2608.27260v1)** `21 · 摄入/运行时检索` — 用「准确性—复杂度—多样性」视角把智能体数据生成形式化为带约束的分布设计问题。
17. **[ASCENT: Online Test-Time Training of Long-Horizon Agents via Self-Distillation of Verified Experience](http://arxiv.org/abs/2610.05303v1)** `20 · 摄入/运行时检索` — 让智能体在部署期间用自己经验证的执行轨迹做自蒸馏，并在线更新权重。
18. **[Agentic Data Engineering for LLMs: System Design and a Controlled Empirical Study](http://arxiv.org/abs/2610.14238v1)** `20 · 摄入/运行时检索` — 检验智能体能否依据实验反馈自主演化数据工程策略，以及这些策略对目标模型的真实增益。
19. **[AutoData: Agentic Search for Pre-training Data Selection](http://arxiv.org/abs/2609.19754v1)** `20 · 摄入/运行时检索` — 让智能体直接在可执行的选择算法空间里搜索，把预训练数据选择变成自动化启发式工程。
20. **[Beyond LLM-Based Reasoning: Lightweight GNNs for Agent Failure Attribution](http://arxiv.org/abs/2608.18575v1)** `20 · 基底/智能体与工具` — 用轻量图网络建模步骤级语义与智能体关系，替代 LLM 判定失败责任方与错误类型。
21. **[Effective Synthetic Data Curation Requires Group-Level Signals](http://arxiv.org/abs/2610.00779v1)** `20 · 创建/数据选择` — 论证合成数据筛选必须看组级信号而非逐条打分，并给出算力预算下优先估计哪些组的廉价诊断。
22. **[Know What You Don't Know: Agentic Data Synthesis by Evolving Generation Space for Closed-Loop Learning](http://arxiv.org/abs/2610.12593v1)** `20 · 创建/合成数据` — 用学习反馈维护可修订的「生成空间状态」，围绕当前学习盲区闭环演化合成数据。
23. **[MiST: Mid-Training LLMs for Cybersecurity](http://arxiv.org/abs/2609.18496v1)** `20 · 摄入/中期训练` — 以中期训练作为通用预训练到网络安全之间的适配阶段，产出 8B/32B 领域模型套件。
24. **[Preference Data Selection for Mitigating the Alignment Tax in Large Language Models](http://arxiv.org/abs/2608.24192v1)** `20 · 基底/指令与偏好` — 用均衡的偏好数据选择抑制灾难性遗忘，在对齐效果与通用能力之间取平衡。
25. **[The Chase Is the Curriculum, the Capture Anchors the Credit: Pursuit-Evasion Self-Play for Zero-Data LLM Reasoning](http://arxiv.org/abs/2608.21871v1)** `20 · 摄入/中期训练` — 把零数据自博弈建模成追逐规避博弈，让出题方沿难度轴持续领先解题方，形成自然课程。
26. **[TrajLong: Co-Designing Agentic and Long-Context Supervision for Mid-Training](http://arxiv.org/abs/2610.04973v1)** `20 · 摄入/中期训练` — 把智能体轨迹编译成带稠密监督的长上下文训练任务，同时锻炼长上下文与原子能力。
27. **[ATLAS-Bench: An Agentic Tool-Use Benchmark for Local Area Search](http://arxiv.org/abs/2610.39438v1)** `19 · 摄入/运行时检索` — 用真实搜索事件扩展的查询与固定 POI 知识库，评测本地搜索是否满足硬性约束与偏好。
28. **[FinSafeGuard: A Quality-Driven Synthetic Dataset for Financial AI Safety](http://arxiv.org/abs/2610.26066v1)** `19 · 创建/合成数据` — 从 63.2 万条场景规格的分类体系出发，四阶段流水线产出 1,430 万条质量过滤的金融安全样本。
29. **[From Trajectories to Grounded Preferences: Process Preference Synthesis via Interaction Element Graphs for Web PRMs](http://arxiv.org/abs/2609.32351v1)** `19 · 摄入/运行时检索` — 用交互元素图引导合成「最小对比对」，缓解网页智能体过程奖励模型的数据稀缺。
30. **[HSRM: Hidden-State Reward Models for Test-Time Verification](http://arxiv.org/abs/2608.30841v1)** `19 · 摄入/运行时验证` — 直接读取生成器内部隐状态验证候选解，省去文本验证器重读整条推理的开销。
31. **[MetaCurate: Evolving Curation Filters for High-Quality Agentic Coding Data](http://arxiv.org/abs/2610.23134v1)** `19 · 基底/智能体与工具` — 免训练地用成对 LLM 评判演化可执行的轨迹筛选规则，选出真正有监督价值的编码轨迹。
32. **[Quality, Not Just Outcome: A Structural Scoring Framework for Selecting Coding-Agent Trajectories](http://arxiv.org/abs/2610.13152v1)** `19 · 基底/智能体与工具` — 直接对轨迹结构做线性时间打分，无需 LLM 裁判即可筛出过程质量高的 SFT 数据。
33. **[RISED: RubrIcs for agentic multi-environment Selection and sElf-Distillation](http://arxiv.org/abs/2610.00979v1)** `19 · 摄入/中期训练` — 用评分细则刻画跨环境关系与组内相对奖励，弥补多环境强化学习中单一标量奖励的信息不足。
34. **[Training-Aware Target Coverage for Synthetic Data Selection](http://arxiv.org/abs/2610.00814v1)** `19 · 创建/合成数据` — 用线性理论刻画合成数据「新增信息 vs 引入误差」的权衡，做随训练集增长而变化的选取。
35. **[Trajectory Soup: Pushing the Compute-Scaling Frontier of LLM Mid-training via Diverse Trajectories](http://arxiv.org/abs/2609.37169v1)** `19 · 摄入/中期训练` — 从共享检查点分叉多条不同配方的训练分支，用兼容多样性突破串行中期训练的收益天花板。
36. **[Agentic Game Development as a Verifiable Trajectory Data Engine for Scaling World Models](http://arxiv.org/abs/2608.25518v1)** `18 · 摄入/运行时检索` — 主张用可执行的游戏开发轨迹提供有接地奖励的递归数据引擎，而非单纯堆爬取视频与算力。
37. **[CodeMidas: Scaling Agentic Coding RL Environments from Code Itself](http://arxiv.org/abs/2609.22068v1)** `18 · 基底/Other` — 仅以源码为任务输入，把既有代码库里已实现的功能自动转成可执行的强化学习环境。
38. **[Creation begins with understanding: LLMs as strategy designers for privacy-preserving tabular data synthesis](http://arxiv.org/abs/2608.29674v1)** `18 · 创建/合成数据` — 让 LLM 设计合成「策略」而非直接生成记录，兼顾表格结构与隐私可审计性。
39. **[DataSense-Bench: The First Step Toward an AI Scientist](http://arxiv.org/abs/2610.12190v1)** `18 · 创建/数据选择` — 用数据选择与性能预测任务，检验前沿模型是否具备「选对训练数据」的能力。
40. **[Drive the Thoughts: Runtime Monitoring of VLA Reasoning-Trajectory Consistency](http://arxiv.org/abs/2608.29583v1)** `18 · 摄入/运行时验证` — 把思维链与动作轨迹的一致性检查嵌入运行时监控，为自动驾驶 VLA 提供可验证性。
41. **[Evaluating Confidence-Gated Retrieval with Matched Trajectory Replay](http://arxiv.org/abs/2608.26846v1)** `18 · 摄入/中期训练` — 用匹配轨迹回放协议，在同一轨迹上公平比较不同「置信度→动作」映射的后果。
42. **[Explorer-Zero: Improving Self-Play for Long-Horizon Tasks with Decomposable Verifiable Rewards](http://arxiv.org/abs/2610.19868v1)** `18 · 摄入/中期训练` — 让自博弈在与环境交互中同时生成任务与细粒度可验证奖励，支撑长时程训练。
43. **[FLARE: A Full-Lifecycle Dense Supervision Paradigm for Long-Horizon Coding Agents via Generative Reward Model](http://arxiv.org/abs/2609.23808v1)** `18 · 摄入/运行时检索` — 用轻量生成式奖励模型提供全生命周期稠密监督，缓解二值奖励的信用分配难题。
44. **[Fisher-Guided Submodular Data Selection for Continual Pre-Training of Large Language Models](http://arxiv.org/abs/2610.02593v1)** `18 · 创建/数据选择` — 揭示基于损失的选择会削弱预训练模型的高 Fisher 坐标，据此用 Fisher 引导的子模选择控制遗忘。
45. **[From Documents to Reasoning: A Validated Synthetic Data Pipeline and Semantic-Aware Fine-Tuning for Financial Numerical Reasoning](http://arxiv.org/abs/2608.27919v1)** `18 · 创建/合成数据` — 用高质量合成数据加 QLoRA 微调小模型，提升跨表格、图表与文本的金融数值推理。
46. **[GTA-RAG: Graph-Trajectory-Augmented Reinforcement Learning for Multi-Turn Retrieval-Augmented Reasoning](http://arxiv.org/abs/2608.22479v1)** `18 · 摄入/运行时验证` — 用图化轨迹增强的强化学习训练多轮「检索—整合证据」的推理策略。
47. **[Industrial-Instruction: An End-to-End Framework for Building Instruction-Tuning and Benchmark Datasets from Industrial Technical Reports](http://arxiv.org/abs/2608.22817v1)** `18 · 创建/合成数据` — 从真实工业技术报告端到端构建指令微调与评测数据集，填补该类文档无公开数据的空白。
48. **[PerceptRL: Making Visual Perception Rewardable in MLLMs from Answers to Interactions](http://arxiv.org/abs/2610.37471v1)** `18 · 摄入/运行时验证` — 从答案层与交互层两级强化细粒度视觉感知，使感知误差能被奖励信号约束。
49. **[ReTAC: Residual Test-Time Agentic Computation for Video Spatial Reasoning](http://arxiv.org/abs/2610.16315v1)** `18 · 摄入/运行时验证` — 只把额外计算分配给基础模型未解决的需求，并在局部失败后仅重算受影响状态。
50. **[Robust LLM Data Selection via Content-Invariant Gradient Attribution](http://arxiv.org/abs/2610.36755v1)** `18 · 创建/数据选择` — 去除梯度归因中的表层形式偏置，让语义等价改写不影响指令数据选择结果。
51. **[SWE-Prime: Fewer Trajectories, Better Performance](http://arxiv.org/abs/2608.27449v1)** `18 · 基底/智能体与工具` — 用轨迹级与片段级两阶段筛选，证明少量高质量 SFT 轨迹胜过大量「成功」轨迹。
52. **[SocialBuddy: Tailoring Search Agent for Social Scenarios](http://arxiv.org/abs/2609.01641v2)** `18 · 摄入/运行时检索` — 首个面向异构用户查询与多维社交信息流的智能体搜索框架。
53. **[Studying Without a Syllabus: Task-Agnostic Environment Preprocessing](http://arxiv.org/abs/2609.10824v1)** `18 · 摄入/运行时检索` — 研究智能体在不知道下游任务分布时，如何自主决定为陌生环境预先构建哪些可复用资源。
54. **[Style-Debiased DPO: Updating LLM Knowledge with Factuality-Aware Synthetic Preference Data](http://arxiv.org/abs/2609.16532v1)** `18 · 基底/指令与偏好` — 用去风格偏置的 DPO 与事实感知偏好对，让模型更好地「取出」而非仅「存入」知识。
55. **[Synthetic Data Characterization via Training Dynamics](http://arxiv.org/abs/2609.39447v1)** `18 · 创建/合成数据` — 用编码器训练动态刻画合成数据与人类数据的样本级可学习性分布及其稳健性。
56. **[Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments](http://arxiv.org/abs/2609.04148v1)** `18 · 基底/智能体与工具` — 把每条终端轨迹转成可反复查询的可执行环境，再从中合成新任务与后续交互。
57. **[The Imitation Game: When LLMs Learn to Reason Like Programs via Code-Centric Reasoning Data Synthesis](http://arxiv.org/abs/2609.16076v1)** `18 · 创建/合成数据` — 用可执行代码作为严格媒介合成推理数据，把自然语言推理锚定到确定性符号执行。
58. **[Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms](http://arxiv.org/abs/2609.27321v1)** `18 · 基底/智能体与工具` — 先定义结果规则再生成状态化环境，比较显式与隐藏参数设定对可验证性的影响。
59. **[Verifier-Guided Self-Consistency: Test-Time Reasoning with Imperfect Generators and Verifiers](http://arxiv.org/abs/2611.06606v1)** `18 · 摄入/运行时验证` — 在解码中结合步骤级正确性与未来结果估计，平衡都不完美的生成器与验证器。
60. **[When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval for LLM Agents](http://arxiv.org/abs/2609.10750v1)** `18 · 创建/合成数据` — 基于 34,396 个技能的生产级路由器，实证合成数据微调提升域内检索却损害域外技能召回。

---

全量 2,807 篇的英文一句话 idea 见 [`PAPERS.md`](PAPERS.md)；自动抽取逻辑见 `scripts/extract_ideas.py`。若需要把中文概括扩展到 score ≥15（160 篇）或 ≥12（443 篇），按同一体例续写即可。

> 补充：本节按**主题相关性**排序（与 `relevance_score` 一致）。若要看按**质量代理分**排序的全量榜单与规则，见 [`PAPER_QUALITY_RANKING.md`](PAPER_QUALITY_RANKING.md) 与 [`config/quality_ranking.yaml`](config/quality_ranking.yaml)。两套分数衡量的是不同东西。
