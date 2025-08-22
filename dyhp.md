data generation

# task data from LMs

## [TarGEN: Targeted Data Generation with Large Language Models](https://arxiv.org/abs/2310.17876)

### Idea

TarGEN 是一个 **多步驱动、无需种子样本** 的高质量合成数据生成框架，专为大型语言模型（LLMs）设计，适用于低资源或新任务场景。
其核心流程如下：

1. **生成语境（Context）**  
   首先生成一个语义场景，如新闻、对话、百科等，为后续样本提供语境支撑。
2. **生成实例种子（Instance Seeds）**  
   在语境中生成一段文本（如一段描述或一个句子），作为样本构造的语义基础。
3. **结合标签约束生成样本（Label-Constrained Generation）**  
   将 instance seeds 与任务定义中的标签（如逻辑蕴含任务中的 `entailment` / `not-entailment`）结合，引导 LLM 生成符合语义约束的训练样本。
4. **零样本生成（Zero-Shot）**  
   整个过程不依赖已有示例，完全基于任务说明和 prompt 完成。

例如，对于文本蕴含任务（RTE），TarGEN 会生成一个前提句子（premise）作为 seed，并根据目标标签生成一个结论句子（hypothesis），使其满足 "蕴含" 或 "不蕴含" 的语义关系。


## [Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation](https://arxiv.org/abs/2402.18334)

### Idea

用 Bonito 将 领域未标注文本 + 任务类型 自动生成高质量的 指令-答案数据，替代人工标注来做指令微调，从而让大模型在新领域任务上实现 零样本适应，并且避免自监督预训练导致的指令遗忘问题。


# from scratch

## [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335)

### Idea

Absolute Zero 提出一种“零数据自进化”的推理模型训练方法，不依赖任何人工构建的题库或标注数据，让同一个模型在自博弈框架中自己出题、自己解题，并用代码执行器验证正确性作为奖励信号，通过强化学习不断提升推理能力。

关键点：

统一模型双角色：同时担任任务提出者（生成可验证推理题）和解答者（解决这些题）。

三类推理任务：基于代码 (程序, 输入, 输出) 设计演绎、溯因、归纳三种核心推理模式。

环境可验证奖励：利用代码执行器自动判断题目合法性和答案正确性，给出稳定的学习信号。

自适应难度：优先生成对当前模型“适中难度”的题，促进高效学习。

完全零人工数据：训练中不依赖人类提供的题目或答案，性能仍超越依赖大规模人工数据的同类方法。


## [Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement](https://arxiv.org/abs/2506.03541#:~:text=In%20this%20paper%2C%20we%20present%20a%20novel%20Debate,error%20analysis%2C%20corrective%20strategies%29%20to%20guide%20student%20models.). Findings of ACL 2025

### Idea

1.提出了 Debate & Reflect (D&R) 框架

小模型（student）和强大的 teacher 模型（GPT-4o, Claude, Gemini 等）进行多轮辩论。

2.提出 Tree-structured Direct Preference Optimization (T-DPO)

将辩论过程记录为 Multi-Agent Interaction Graph (MAG)，再转化为 偏好树 (Preference Trees)。

每个树根节点包含问题及上轮的 structured 信息，分支节点是正确和错误的回答。

学生通过 T-DPO 学到的不仅是 哪个答案是正确的，而且是 为什么某些答案更优，也就是吸收正确推理路径，同时避免重复错误推理。


## [Synthetic Data RL: Task Definition Is All You Need](https://arxiv.org/abs/2505.17063)
### Idea

从任务定义（包括任务描述、输入输出格式）出发，提取关键词，检索相关外部知识（如 Wikipedia、StackExchange）。

使用一个更强的 instructor LLM（如 GPT-4o）基于这些知识生成高质量的合成数据（Q-A 对）。

完全不需要人工标注，只要一份任务定义即可。

在 GSM8K、MATH、MedQA、CQA、CFA 等任务上，表现超过了指令微调 (instruction-tuned) 和其他合成数据方法，甚至接近或超过使用全量人工数据的 RL。


## [Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models](https://arxiv.org/abs/2402.13064)
### Idea

不依赖种子数据，完全从0开始
先由 LLM（GPT-4）和少量人工校验构建 人类知识与能力的分类树（taxonomy），覆盖各个学科和技能领域，最终得到覆盖广泛、难度多样的 指令调优数据。


## [Self-instruct: Aligning language models with self-generated instructions](https://arxiv.org/abs/2212.10560)
### Idea
需要少量的seed tasks（175个）
根据instructio生成input first/ output first两类数据后加入task pool 自举式的扩充种子数据池（自己迭代）

## [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)
### Idea
对齐后的 LLM（如 Llama-3-Instruct）在输入“仅有 pre-query 模板（用户输入开头标记）”时，会因为自回归特性自动生成一条高质量、多样化的“用户问题/指令”。
不依赖人工构造或少量种子问题，而是直接利用 LLM 内隐学到的“指令分布”。
利用这个特点，可以自动的生成instruction，并生成回答，构造训练数据；

## [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)
### Idea
提出 Evol-Instruct，将简单指令逐步演化为更复杂的指令：加约束、增加推理步骤、细化概念、复杂化输入等
从一小部分初始指令出发；
通过 ChatGPT 等大模型多轮迭代演化，生成海量不同难度的指令及对应答案；
将这些数据用来微调开源 LLaMA 模型，得到 WizardLM。

SELF-ALIGNMENT WITH INSTRUCTION BACKTRANSLATION

CodecLM: Aligning Language Models with Tailored Synthetic Data

Scaling Synthetic Data Creation with 1,000,000,000 Personas

Condor: Enhance LLM Alignment with Knowledge-Driven
Data Synthesis and Refinement

SAND-Math: Using LLMs to Generate Novel,
Difficult and Useful Mathematics Questions
and Answers
