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
