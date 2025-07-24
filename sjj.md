1. ProtoReasoning: Prototypes as the Foundation for Generalizable Reasoning in LLMs https://arxiv.org/abs/2506.15211

   将问题采用原型表征进行微调训练，用大模型将问题转换为Prolog/PDDL格式，并用对应的Prolog/PDDL解释器来验证数据集构建推理是否正确。

2. Faithful Logical Reasoning via Symbolic Chain-of-Thought

   tranlator将问题和陈述转换成符号化，接着planer提出一系列推理见解。solver提供推理步骤，verifer验证以上步骤。

3. Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework

   Logical Decomposer (逻辑分解器):将句子进行标准化分解。逻辑搜索路由：将句子中找到互补的句子。

   逻辑解决器：接收 C_current 和 C_complement，消去互补文字，将剩余文字析取连接，生成新的解决子句 C_resolved

   如果 C_resolved 是空子句（矛盾 ⊥）或确认无矛盾，则终止当前路径推理。

   简单来说就是通过相反方向去同时推理，如果推理到最后或者过程中相互矛盾就是有问题。

4. Symbolic Regression with a Learned Concept Library

   建立一个可解释的抽象概念库C，例如：“表达式中存在三角函数（如 sin），表明与波形或周期性物理现象有关”，或“变量间表现出幂律关系”。

   概念指导的假设演化 (Hypothesis Evolution):从概念库C对数据通过变异和交叉组合。利用 LLM 的背景知识和上下文学习能力，将人类科学家的“直觉指导”引入到进化搜索中，引导搜索向更可能包含正确抽象结构的方向发展。

   概念抽象 (Concept Abstraction):高表现的假设加入到C库中。

   概念进化 (Concept Evolution):新生成的概念也加入到C中，无论是否验证了对错（后期更多探索途径）

5. Large Language Models Are Neurosymbolic Reasoners

   LLM 可以在外部符号模块（如计算器、导航器）的辅助下，通过精心设计的提示（Prompting），有效地扮演神经符号推理器（Neurosymbolic Reasoner）的角色，解决需要符号操作的任务，而无需依赖大量的专家数据或复杂的强化学习训练。

6. A Survey on Enhancing Large Language Models with Symbolic Reasoning

   针对LLMs在自然语言推理中存在的语义模糊性和逻辑不一致问题（如幻觉），论文提出**引入符号语言**（逻辑语言、编程语言、数学表达式等）可提升推理的精确性和可靠性。核心框架是让LLMs专注于将问题转化为**形式化的符号表示**（如一阶逻辑、Python代码、数学方程），再交由外部求解器执行计算，实现互补协作。论文从四个维度展开：

   1. **任务场景**：详述符号推理在逻辑推理（分解问题+外部求解器）、数学推理（代码生成+验证）、表格推理（SQL/Python分解）、空间推理（符号化提示）及规划任务（PDDL语言）中的应用；
   2. **符号语言类型**：分类探讨逻辑符号（如Prolog）、编程语言（如Python）、数学符号及领域专用语言（如ASP、PDDL）的适用性；
   3. **增强技术**：总结微调（专用数据集训练）、提示工程（符号化思维链）及混合方法的优劣；
   4. **评估基准**：整理各任务的主流数据集（如GSM8K数学推理、ProofWriter逻辑推理）。文末指出未来方向包括**定制化符号语言设计**、**多符号语言融合**及赋予LLMs**内在结构化推理能力**，以突破现有局限。

7. LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning

   这篇论文提出了 **Logic-LM 框架**，通过将大语言模型（LLMs）与符号求解器结合，显著提升复杂逻辑推理的准确性和可靠性。其核心思想是**分阶段处理逻辑问题**：

   1. **问题形式化**：LLM 将自然语言问题转化为结构化符号表示（如逻辑编程语言 Prolog、一阶逻辑 FOL、约束满足问题 CSP 或布尔可满足性问题 SAT）；
   2. **符号推理**：调用外部符号求解器（如 Pyke、Prover9、Z3）执行确定性推理；
   3. **结果解释**：将符号结果转译为自然语言答案。

   创新点包括：

   - **自优化模块**：利用求解器的错误反馈迭代修正符号表达，提升形式化准确性；
   - **多任务适配**：针对五类逻辑问题（演绎推理、FOL、CSP、分析推理）定制符号语言与求解器，覆盖 PrOntoQA 等五大基准数据集。

8. Com²: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models

   这篇论文提出了 **Com²**，一个专门用于评估大型语言模型（LLMs）在**复杂常识推理**方面能力的基准。针对现有基准在复杂、隐含且常涉及长期影响或罕见场景的常识推理任务上的不足，Com² 利用**因果事件图（CEG）** 结构化地表示复杂常识知识，并应用**因果理论（如干预、反事实）** 修改 CEG 来创建符合人类关切的不同推理场景（如直接、决策、过渡、干预、反事实）。随后，基于修改后的 CEG，通过 LLM 的“慢思考”生成包含多选/多选问题的数据集（Com²-main）。此外，还构建了更具挑战性的子集 **Com²-hard**，基于需要复杂线索组合推理的侦探故事。实验评估了多种 LLM，发现它们在推理深度（处理长因果链）和广度（处理罕见/突发场景）上存在局限，并表明**后训练（post-training）** 和提供**慢思考（slow thinking）** 指导能有效缓解这些不足。Com² 旨在填补复杂常识推理评估的空白，为开发更强大的 LLM 提供洞见。

9. Towards Reasoning in Large Language Models: A Survey

   这篇综述论文系统梳理了大语言模型（LLM）推理能力的研究现状。论文首先澄清了推理的概念（如演绎、归纳、溯因推理）及其在LLM语境下的含义（主要指非形式化演绎推理）。核心部分详细评述了提升和激发LLM推理的技术，重点包括：1) **提示工程**，如思维链（CoT）及其变体（零样本CoT、自洽性解码）、问题分解（最小到最多提示）和理由工程（优化、探索、验证）；2) **混合方法**，如结合推理数据进行微调或预训练，以及模型自我提升（如STaR）。论文还总结了评估LLM推理能力的方法与常用基准（如算术推理的GSM8K、常识推理的CSQA、符号推理任务），并分析了现有评估的局限性（侧重最终任务性能而非推理过程质量）。关键发现指出：推理能力似乎是LLM在超大规模（如百亿参数以上）时涌现的特性；CoT提示能有效激发LLM展现类似推理的行为并提升鲁棒性；LLM会表现出类似人类的推理偏差；但LLM在复杂推理任务上仍表现不佳，且其“推理”本质（是基于真实逻辑还是启发式模式）仍存在巨大争议，现有证据不足以断定LLM具备真正的推理能力。最后，论文讨论了该领域的意义、挑战和未来方向，强调需要开发更严格的评估方法、研究更现实的应用场景，并深入探索如何从根本上提升LLM的推理能力。
