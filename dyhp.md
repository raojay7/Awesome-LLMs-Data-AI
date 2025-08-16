data generation

# task data from LMs

## TarGEN: Targeted Data Generation with Large Language Models

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


## Learning to Generate Instruction Tuning Datasets for

### Idea

用 Bonito 将 领域未标注文本 + 任务类型 自动生成高质量的 指令-答案数据，替代人工标注来做指令微调，从而让大模型在新领域任务上实现 零样本适应，并且避免自监督预训练导致的指令遗忘问题。


# from scratch

## Absolute Zero: Reinforced Self-play Reasoning with Zero Data

### Idea

Absolute Zero 提出一种“零数据自进化”的推理模型训练方法，不依赖任何人工构建的题库或标注数据，让同一个模型在自博弈框架中自己出题、自己解题，并用代码执行器验证正确性作为奖励信号，通过强化学习不断提升推理能力。

关键点：

统一模型双角色：同时担任任务提出者（生成可验证推理题）和解答者（解决这些题）。

三类推理任务：基于代码 (程序, 输入, 输出) 设计演绎、溯因、归纳三种核心推理模式。

环境可验证奖励：利用代码执行器自动判断题目合法性和答案正确性，给出稳定的学习信号。

自适应难度：优先生成对当前模型“适中难度”的题，促进高效学习。

完全零人工数据：训练中不依赖人类提供的题目或答案，性能仍超越依赖大规模人工数据的同类方法。


## Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement. Findings of ACL 2025

### Idea

1.提出了 Debate & Reflect (D&R) 框架

小模型（student）和强大的 teacher 模型（GPT-4o, Claude, Gemini 等）进行多轮辩论。

2.提出 Tree-structured Direct Preference Optimization (T-DPO)

将辩论过程记录为 Multi-Agent Interaction Graph (MAG)，再转化为 偏好树 (Preference Trees)。

每个树根节点包含问题及上轮的 structured 信息，分支节点是正确和错误的回答。

学生通过 T-DPO 学到的不仅是 哪个答案是正确的，而且是 为什么某些答案更优，也就是吸收正确推理路径，同时避免重复错误推理。
