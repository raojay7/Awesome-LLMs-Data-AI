1. ProtoReasoning: Prototypes as the Foundation for Generalizable Reasoning in LLMs

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


