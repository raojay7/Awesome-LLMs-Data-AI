
Wang, S., Jin, X., Wang, Z., Wang, J., Zhang, J., Li, K., Wen, Z., Li, Z., He, C., Hu, X., Zhang, L., 2025. Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning. https://doi.org/10.48550/arXiv.2505.12212
  
Data Whisperer 提出一种**无需额外训练、基于注意力权重**的数据选择方法，利用目标 LLM 自身的少样本上下文学习（ICL）能力，通过随机采样少量演示-查询对评估每条训练样本的贡献度，并以注意力交互强度为权重消除顺序敏感，最终选出 1%–10% 的高价值子集，在多个任务上达到或超越全数据微调性能，且速度提升 7–20 倍；还可借助同族小模型实现弱-强扩展，显著降低计算成本。  

Zhang, J., Zhang, C.-X., Liu, Yao, Jin, Y.-X., Yang, X.-W., Zheng, B., Liu, Yi, Guo, L.-Z., 2025. D3: Diversity, Difficulty, and Dependability-Aware Data Selection for Sample-Efficient LLM Instruction Tuning. https://doi.org/10.48550/arXiv.2503.11441
  
D3 提出一套**“多样性-难度-可信度”三维联合的数据选择框架，用于在**极少量样本（5%-10%）**条件下实现高效、鲁棒的 LLM 指令微调。首先，通过**样本间多样性**（k-center 距离）、**模型-样本难度**（基于熵修正的 UPD 指标）以及**外部教师模型给出的可信度**三重评分，D3 对大规模指令数据集进行系统性评估；随后将三者集成到一个加权核集优化目标中，以贪心迭代方式选出最具价值的核心子集。该框架支持**多轮反馈迭代**，可随微调进程自适应调整数据焦点。在 Alpaca、WizardLM 等公开数据集及淘宝直播真实场景中，D3 仅用 5%-10% 的数据即可达到甚至超越全数据微调效果，显著优于随机、PPL、IFD 等基线，验证了其兼顾**效率、性能与鲁棒性**的优势。  

Zeng, X., Wang, H., Lin, J., Wu, J., Cody, T., Zhou, D., 2025. LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection. https://doi.org/10.48550/arXiv.2505.03793
 
LENSLLM 提出了一套**理论驱动、计算高效**的 LLM 选择框架，通过**PAC-Bayesian 泛化界**首次刻画了微调过程中的“预幂期–幂期”双阶段动态，并借助**神经正切核（NTK）**构建可解释的缩放模型。其核心思想是用 NTK 捕获模型在极小训练集上的损失曲线，再外推到全量数据场景，从而在不完整训练的情况下预测最终性能。实验覆盖 14 种跨规模模型与 3 个基准任务，结果显示 LENSLLM 在**皮尔逊相关性（最高 85.8%）和相对准确率（最高 91.1%）**上均大幅超越 5 个 SOTA 基线，同时将算力成本降低 **88.5%**。此外，渐进式采样与双阶段停止策略使其在超参数、输入长度变化下保持鲁棒，为资源受限场景下的通用 LLM 选择提供了可靠的理论与工程范式。  

Yang, Y., Nan, Y., Ye, J., Dou, S., Wang, X., Li, S., Lv, H., Wu, M., Gui, T., Zhang, Q., Huang, X., 2025. Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric. https://doi.org/10.48550/arXiv.2502.17184
 
提出 NovelSum 指标，用“邻近加权+密度感知”衡量样本新颖度，与 LLM 指令微调性能相关 0.97；配套贪婪策略 NovelSelect 选 10 k 样本即优于现有方法。  

Wettig, A., Gupta, A., Malik, S., Chen, D., 2024. QuRating: Selecting High-Quality Data for Training Language Models. https://doi.org/10.48550/arXiv.2402.09739

QuRating 提出一种基于大模型 pairwise 质量判别（写作风格、事实密度、教育价值、所需专业知识）的数据筛选框架，用 Bradley-Terry 模型把判别转为标量分数，再以温度采样选取 30 B token 训练 1.3 B 模型，结果仅用一半算力就能达到均匀采样的效果，并公开发布了 QuRatedPajama 数据集与 QuRater 模型。  

Wang, Z., Zhu, Q., Mi, F., Xu, M., Jin, R., Yang, W., 2025. ClusterUCB: Efficient Gradient-Based Data Selection for Targeted Fine-Tuning of LLMs. https://doi.org/10.48550/arXiv.2506.10288

ClusterUCB 先用梯度余弦相似度对训练数据聚类，再把簇间选择建模为多臂老虎机，用改进 UCB 在有限计算预算内高效采样高影响样本；实验表明仅用 20% 梯度计算即可达到原梯度法效果，显著降低开销。
