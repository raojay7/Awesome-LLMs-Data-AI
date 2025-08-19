## 综述--关于大模型在数据合成、符号推理等方面的综述性论文。

1. [Towards Reasoning in Large Language Models: A Survey](https://arxiv.org/abs/2212.10403)

   这篇综述论文系统梳理了大语言模型（LLM）推理能力的研究现状。论文首先澄清了推理的概念（如演绎、归纳、溯因推理）及其在LLM语境下的含义（主要指非形式化演绎推理）。

2. [A Survey on Enhancing Large Language Models with Symbolic Reasoning](https://openreview.net/forum?id=exg4ByWdrM)

   一篇发表于 2025 年的综合性综述，系统梳理了 LLM 在符号推理方面的研究进展，包括：自动生成结构化推理数据；符号逻辑系统如何与生成模型结合；方法分类、系统挑战与未来发展方向。

3. [A Survey on Symbolic Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2408.10210)

   该文章集中讨论如何提取 LLM 中的隐性知识为可解释的 **符号形式**，强化模型可解释性与推理能力.包括方法归类与关键挑战分析；针对 Symbolic Knowledge 的蒸馏流程做系统梳理.

4. [Neurosymbolic AI for Reasoning over Knowledge Graphs: A Survey](https://arxiv.org/abs/2302.07200)

   知识图谱的推理任务介绍

5. [Neuro-Symbolic AI in 2024: A Systematic Review](https://arxiv.org/abs/2501.05435)

   系统收录了 2020–2024 年间关于符号推理的 1,428 篇文献，最终精选 167 篇进行深入分析。

6. [Preserving Privacy in Large Language Models: A Survey on Current Threats and Solutions](https://arxiv.org/abs/2408.05212) *arxiv 2024* 

   系统综述了大语言模型的隐私风险与保护方法，涵盖成员推理攻击、训练数据泄露、差分隐私训练与私合成数据生成等。文章从攻击、防护、数据删除三个维度进行分类。为研究者提供了全面的知识图谱和未来方向。  

7. [Privacy-Preserving Large Language Models: Mechanisms and Applications](https://arxiv.org/html/2412.06113v1) *arxiv 2024* 

   总结LLM隐私保护机制，包括差分隐私、联邦学习、加密推理和私有数据合成等。重点讨论如何在医疗、法律等应用中实际部署私合成数据方案。提供了实证隐私测试方法和实际应用案例。  

## 推理数据合成--关于如何合成符号数据的论文

1. [Neuro‑Symbolic Data Generation for Math Reasoning (NeurIPS 2024)](https://arxiv.org/abs/2412.04857)

   提出一种 神经符号混合（neuro‑symbolic）框架，用于自动合成数学推理训练数据。该方法仔细地改变现有的数学问题，确保新生成的问题的多样性和有效性。这是通过神经符号数据生成框架实现的，该框架结合了LLMS的直观非形式化优势、数学求解器的精确符号推理以及高度不规则符号空间中的投影马尔可夫链蒙特卡洛采样。

2. [Generating Data for Symbolic Language with Large Language Models (SymGen)](https://aclanthology.org/2023.emnlp-main.523.pdf) *EMNLP 2023*

    提出SymGen框架，用大语言模型生成符号语言（如逻辑表达式、代码）数据，并结合一致性校验提升质量。实验表明合成数据能显著增强小模型在语义解析和逻辑推理上的表现。该方法展示了LLM在符号任务数据合成上的潜力。

   

## 隐私保护--关于对大模型合成数据进行去敏感化数据合成

*以差分隐私为主流，通过加入噪声数据来保护隐私信息。*

1. [SafeSynthDP: Leveraging Large Language Models for Privacy-Preserving Synthetic Data Generation Using Differential Privacy](https://arxiv.org/abs/2412.20641)

   研究了大型语言模型（LLMs）生成与差分隐私（DP）机制集成的合成数据集的能力，从而实现数据驱动的研究和模型训练，而无需直接暴露敏感信息。我们的方法将基于 DP 的噪声注入方法（包括拉普拉斯分布和高斯分布）纳入数据生成过程。

2. [Enhancing Leakage Attacks on Searchable Symmetric Encryption Using LLM‑Based Synthetic Data Generation ](https://www.arxiv.org/abs/2504.20414)

   提出了一种新颖的方法，利用大型语言模型 （LLM），特别是 GPT-4 变体，生成在统计和语义上类似于安然电子邮件真实世界数据集的合成文档。

3. [HARMONIC: Harnessing LLMs for Tabular Data Synthesis and Privacy Protection](https://arxiv.org/abs/2408.02927)

   通过引入用于表格数据生成和评估的新框架 HARMONIC，探索用于表格数据合成和隐私保护的 LLM。

4. [Model-based Large Language Model Customization as Service](https://arxiv.org/abs/2410.10481)

   引入了 Llamdex，这是一个新颖的框架，可以促进 LLM 定制即服务，客户端在其中上传预训练的特定领域模型而不是数据。这个客户端上传的模型，可选地由 DP 保护，噪声低得多，通过连接模块插入到基本 LLM 中。

5. [Towards Verifiable Text Generation with Symbolic References](https://arxiv.org/abs/2311.09188)

   虽不直接处理隐私问题，但该论文提出了 SymGen 框架，在 LLM 输出中嵌入符号引用，便于追溯生成内容与原始数据之间的对应关系，有助于验证与可控性，是符号化生成与安全控制结合的一个方向。SymGen 提示 LLM 将其常规输出文本与对某些条件数据（例如，JSON 格式的表）中存在的字段的显式符号引用交错。引用可用于显示生成中不同跨度文本的出处，从而减少手动验证所需的工作量。

6. [Generative Private Synthetic Data via Foundation Model APIs 2: Text](https://icml.cc/virtual/2024/poster/34291) *ICML 2024* 

   提出Aug-PE方法，仅通过调用LLM API而非私有微调即可生成差分隐私保证的合成文本。该方法在不需要访问专有模型参数的情况下实现了正式的DP理论保证，并在多个NLP任务中接近甚至超过DP-SGD微调效果。适用于机构只能API级调用大模型的隐私敏感场景。 

7.  [Private Text Generation by Seeding Large Language Model Prompts](https://openreview.net/pdf?id=rw25QGrkNy) *NeurIPS 2024* 

   提出DP-KPS方法，将敏感文本嵌入为差分隐私保护的关键词，再作为提示词种子驱动LLM生成。这样避免了直接暴露私有语料，同时保持较高的下游任务性能。该框架特别适合医疗、金融等场景中的安全语料合成。  

8.  [SeqPATE: Differentially Private Text Generation via PATE](https://proceedings.neurips.cc/paper_files/paper/2022/file/480045ad846b44bf31441c1f1d9dd768-Supplemental-Conference.pdf) *NeurIPS 2022* 

   将教师-学生式的PATE框架扩展到序列文本生成。通过多个教师模型投票并加入噪声，保证生成文本满足差分隐私约束。实验表明在NLP任务中能有效平衡隐私与效用

9.  [Harnessing Large-Language Models to Generate Private Synthetic Text](https://openreview.net/forum?id=TOE6N8dp4w) *ICLR 2024* 

   系统研究如何利用LLM生成差分隐私的合成数据，并用这些数据训练下游模型。论文比较了不同DP机制与合成策略，发现私合成数据在部分任务能接近真数据效果。该工作为DP合成的实证研究提供了全面基准。  

10.  [KnowledgeSG: Privacy-Preserving Synthetic Text Generation with Knowledge Distillation from Server](https://aclanthology.org/2024.emnlp-main.438/) *EMNLP 2024*

提出KnowledgeSG框架，将客户端侧的DP训练与服务器侧的知识蒸馏结合。这样既能保护用户隐私，又能提升合成数据的可用性。实验显示在医疗、金融等敏感领域下游任务中优于传统DP生成方法。

12. [Evaluating Differentially Private Synthetic Data Generation for NLP](https://aclanthology.org/2024.findings-emnlp.894.pdf) *EMNLP 2024 Findings* 

    系统性评估了多种差分隐私文本合成方法在NLP任务中的表现。研究指出现有方法在高风险场景下存在显著效用损失，并提出改进的评测指标。为未来私合成研究提供了风险警示和改进方向。  

13. [Privacy-Preserving Synthetic Data Generation for Recommendation Systems (UPC-SDG)](https://arxiv.org/abs/2209.13133) *SIGIR 2022* 

    提出UPC-SDG框架，允许用户在合成数据中设定隐私保护级别。方法兼顾了推荐系统中的数据效用与差分隐私约束，并在真实推荐数据集上取得良好效果。为用户侧可控的私有合成开辟了新方向。

    

## 符号推理思维链框架方法--关于利用大语言模型如何构建符号推理框架的论文

*核心点在于如何验证大模型的推理，利用外部工具或者设计一种有效的思维链来验证推理过程，然后迭代优化*

1. [ProtoReasoning: Prototypes as the Foundation for Generalizable Reasoning in LLMs](https://arxiv.org/abs/2506.15211)

   将问题采用原型表征进行微调训练，用大模型将问题转换为Prolog/PDDL格式，并用对应的Prolog/PDDL解释器来验证数据集构建推理是否正确。

2. [Faithful Logical Reasoning via Symbolic Chain-of-Thought](https://arxiv.org/abs/2405.18357)

   tranlator将问题和陈述转换成符号化，接着planer提出一系列推理见解。solver提供推理步骤，verifer验证以上步骤。

3. [Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework](https://arxiv.org/abs/2412.16953)

   一种可以验证推理过程的推理框架。Logical Decomposer (逻辑分解器):将句子进行标准化分解。逻辑搜索路由：将句子中找到互补的句子。

   逻辑解决器：接收 C_current 和 C_complement，消去互补文字，将剩余文字析取连接，生成新的解决子句 C_resolved

   如果 C_resolved 是空子句（矛盾 ⊥）或确认无矛盾，则终止当前路径推理。

   简单来说就是通过相反方向去同时推理，如果推理到最后或者过程中相互矛盾就是有问题。

4. [Symbolic Regression with a Learned Concept Library](https://arxiv.org/abs/2409.09359)

   建立一个可解释的抽象概念库C，例如：“表达式中存在三角函数（如 sin），表明与波形或周期性物理现象有关”，或“变量间表现出幂律关系”。概念指导的假设演化 (Hypothesis Evolution):从概念库C对数据通过变异和交叉组合。利用 LLM 的背景知识和上下文学习能力，将人类科学家的“直觉指导”引入到进化搜索中，引导搜索向更可能包含正确抽象结构的方向发展。概念抽象 (Concept Abstraction):高表现的假设加入到C库中。概念进化 (Concept Evolution):新生成的概念也加入到C中，无论是否验证了对错（后期更多探索途径）

5. [Large Language Models Are Neurosymbolic Reasoners](https://arxiv.org/abs/2401.09334)

   LLM 可以在外部符号模块（如计算器、导航器）的辅助下，通过精心设计的提示（Prompting），有效地扮演神经符号推理器（Neurosymbolic Reasoner）的角色，解决需要符号操作的任务，而无需依赖大量的专家数据或复杂的强化学习训练。

6. [LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning](https://arxiv.org/abs/2305.12295)

   这篇论文提出了 Logic-LM 框架，通过将大语言模型（LLMs）与符号求解器结合，显著提升复杂逻辑推理的准确性和可靠性。其核心思想是分阶段处理逻辑问题：LLM 将自然语言问题转化为结构化符号表示（如逻辑编程语言 Prolog、一阶逻辑 FOL、约束满足问题 CSP 或布尔可满足性问题 SAT）,调用外部符号求解器（如 Pyke、Prover9、Z3）执行确定性推理；将符号结果转译为自然语言答案。

7. [Com²: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models](https://arxiv.org/abs/2506.07064)

   这篇论文提出了 Com²，一个专门用于评估大型语言模型（LLMs）在复杂常识推理方面能力的基准。针对现有基准在复杂、隐含且常涉及长期影响或罕见场景的常识推理任务上的不足，Com² 利用因果事件图（CEG） 结构化地表示复杂常识知识，并应用因果理论（如干预、反事实） 修改 CEG 来创建符合人类关切的不同推理场景（如直接、决策、过渡、干预、反事实）。随后，基于修改后的 CEG，通过 LLM 的“慢思考”生成包含多选/多选问题的数据集（Com²-main）。此外，还构建了更具挑战性的子集 Com²-hard，基于需要复杂线索组合推理的侦探故事。实验评估了多种 LLM，发现它们在推理深度（处理长因果链）和广度（处理罕见/突发场景）上存在局限，并表明后训练（post-training） 和提供慢思考（slow thinking） 指导能有效缓解这些不足。Com² 旨在填补复杂常识推理评估的空白，为开发更强大的 LLM 提供洞见。

8. [Sound and Complete Neurosymbolic Reasoning with LLM-Grounded Interpretations](https://arxiv.org/abs/2507.09751)

   试图解决如何利用大型语言模型（LLMs）的知识进行形式逻辑推理的问题，同时克服LLMs输出中固有的逻辑不一致性和不完整性

9. [SymbolicThought: Integrating Language Models and Symbolic Reasoning for Consistent and Interpretable Human Relationship Understanding](https://arxiv.org/abs/2507.04189)

   SymbolicThought是一个人机交互框架，它将基于 LLM 的提取与符号推理相结合。该系统构建可编辑的字符关系图，使用七种类型的逻辑约束对其进行细化，并通过交互式界面实现实时验证和冲突解决。

10. [StrucText-Eval: Evaluating Large Language Model's Reasoning Ability in Structure-Rich Text](https://arxiv.org/abs/2406.10621)

    提出了一种自动评估数据生成方法，用于评估LLMS对富结构文本的推理能力，以对此进行探索。支持 8 种结构化语言和 29 个任务，通过可控的嵌套和结构宽度生成复杂度可调的数据。

11. [LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](https://arxiv.org/abs/2409.12929)

    提出了一种名为 LogicPro 的新数据合成方法，该方法利用 LeetCode 风格的算法 Problems 及其对应的Program 解决方案，以文本格式合成复杂的 Logic 推理数据。

12. [iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](https://arxiv.org/abs/2506.01784)

     iQUEST是一个问题引导的 KBQA 框架，它将复杂的查询迭代分解为更简单的子问题，确保结构化和集中的推理轨迹。

## 多语言推理--大语言模型在多语言方向进行推理的论文

*将语言的内容逻辑抽象出来再进行有效推理*

1. [NeuroSymbolic Augmented Reasoning (NSAR)](https://arxiv.org/abs/2506.02483)

   大型语言模型 通常难以在相关信息分散在大量文档中的长上下文场景中执行多目标推理。为了应对这一挑战，引入了神经符号增强推理 （NSAR），它结合了推理过程中神经推理和符号推理的优势。NSAR 从文本中显式提取符号事实并生成可执行的 Python 代码来处理复杂的推理步骤。

2. [AdaMCoT: Rethinking Cross-Lingual Factual Reasoning through Adaptive Multilingual Chain-of-Thought](https://arxiv.org/abs/2501.16154)

   针对跨语言事实推理，动态选择中间“思考语言”以增强低资源语言的一致性与准确性。了 AdaMCOT（自适应多语言思维链），这是一个框架，它通过在生成目标语言响应之前动态路由中间“思维语言”中的思维过程来增强多语言事实推理。

3. [PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts](https://arxiv.org/abs/2504.18428)

   PolyMath是一个多语言数学推理基准测试，涵盖18种语言和4个从易到难的难度级别。我们的基准确保了难度的全面性、语言多样性和高质量的翻译，使其成为推理LLMS时代极具辨别力的多语言数学基准。

4. [Scaling Synthetic Logical Reasoning Datasets with Context-Sensitive Declarative Grammars](https://arxiv.org/abs/2406.11035)*Sina Bagheri Nezhad , Ameeta Agrawal。 arxiv2025*

   提出了一个更简单、更通用的声明式框架，具有灵活的上下文相关规则，绑定多种语言（特别是简体英语和 TPTP 定理证明语言）。我们通过选择多达 32 个前提和一个假设来构建一阶逻辑问题。

5. [MURI: High-Quality Instruction Tuning Datasets for Low-Resource Languages via Reverse Instructions](https://arxiv.org/abs/2409.12958)

   提出了一种新颖的方法——多语言逆向指令（Multilingual Reverse Instructions, MURI）。该方法能够在不需要人工标注者或预先存在的多语言模型的情况下，为低资源语言生成高质量的指令微调数据集。MURI 利用逆向指令和一个翻译pipeline，从低资源语言中现有的人类书写文本中生成指令-输出对。

## 思维链推理方法--聚焦用模型自生成数据（指令、推理过程、偏好/反馈、树搜索轨迹等）来提升推理能力

1. [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)  *NeurIPS 2022*  
   提出一种自举方法：模型先自生成带有推理步骤的解答，再基于正确解答反向训练自身，从而不断迭代提升推理能力。实验表明在算术、逻辑推理任务中，合成推理链能够大幅提高准确率。

2. [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://aclanthology.org/2023.acl-long.754.pdf)  *ACL 2023*  
   提出 **Self-Instruct 框架**：利用模型自身生成指令-输入-输出三元组，并经过自动过滤与去重，构造大规模合成指令数据集进行微调。该方法显著减少人工标注依赖，推动了指令微调的发展。

3. [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)*ICLR 2024*  

   探讨如何利用教师大模型生成的偏好反馈替代人工标注，提出 **RLAIF** 框架。实验表明，纯 AI 合成的偏好数据在摘要与开放对话任务上接近甚至超过 RLHF 效果，证明了大规模可扩展性。

4. [https://arxiv.org/abs/2212.08073](https://arxiv.org/pdf/2212.08073)  *NeurIPS 2023*  

   提出 **宪法式 AI 对齐方法**：通过“宪法原则”指导模型自评与自修正，自动合成安全偏好数据，再进行监督微调和RL阶段，减少了人工介入，显著提升了模型的安全性与合规性。

5. [Chain-of-Preference Optimization (CPO)](https://proceedings.neurips.cc/paper_files/paper/2024/file/00d80722b756de0166523a87805dd00f-Paper-Conference.pdf)  *NeurIPS 2024*  
   提出 **CPO**：基于 Tree-of-Thought 搜索生成多条推理路径，并构造步骤级偏好信号，引导模型学习更优的推理链，解决了传统偏好对齐方法无法利用中间推理步骤的不足。

6. [Chain-of-Thought Reasoning Without Prompting](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7a8e7fd295aa04eac4b470ae27f8785c-Abstract-Conference.html)  *NeurIPS 2024*  
   探索无需显式提示就能触发模型内部推理链的机制，通过采样与解码策略挖掘潜在推理分支，并生成高置信度的合成推理样本，进一步提升了蒸馏与再训练效果。

7. [Knowledge-Augmented Reasoning Distillation](https://neurips.cc/virtual/2023/poster/70015)  *NeurIPS 2023*  
   提出一种知识增强推理蒸馏方法：结合教师模型的推理链与外部知识信号，对小模型进行多层次蒸馏，在知识密集型推理任务中表现显著提升。

8. [Dialogue Chain-of-Thought Distillation for Commonsense Reasoning](https://aclanthology.org/2023.emnlp-main.342)  *EMNLP 2023*  
   提出在对话形式下生成 Chain-of-Thought，并将其蒸馏到较小的对话模型，显著提升了常识推理能力与对话一致性，展示了合成对话推理链的价值。

9. [On-Policy Distillation of Language Models](https://iclr.cc/media/iclr-2024/Slides/19484)  *ICLR 2024*  
   提出 **在策略蒸馏** 方法：学生模型先生成轨迹，教师模型再对这些合成数据提供评分或分布，学生最小化差异。避免了单纯依赖离线数据，提高了生成式推理任务中的泛化。

10. [NoRa: Chain-of-Thought Reasoning with Noisy Rationales](https://neurips.cc/virtual/2024/poster/95956)  *NeurIPS 2024*  
    系统研究了噪声推理链对模型训练的影响，并提出稳健化方法，有效过滤和利用低质量合成CoT数据，为大规模自动生成推理链提供了质量控制机制。

11. [Iter-CoT: Iterative Bootstrapping in Chain-of-Thoughts Prompting](https://aclanthology.org/2024.findings-naacl.257)  *Findings of NAACL 2024*  
    提出迭代自举方法：模型不断生成推理示例，并通过难度控制与自纠错机制筛选更优样本，逐步扩展高质量的推理数据集，提升了推理蒸馏的稳定性。




