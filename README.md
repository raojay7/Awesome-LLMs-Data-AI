# Awesome-LLMs-Data-AI
# A Survey on Data-Centric LLMs Lifecycle

<div align="center">

[![LICENSE](https://img.shields.io/github/license/wasiahmad/Awesome-LLM-Synthetic-Data-Generation)](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data-Generation/blob/main/LICENSE)
![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
<!-- ![license](https://img.shields.io/bower/l/bootstrap?style=plastic) -->

</div>

This repo includes papers and blogs about the survey on Data-centric AI of LLMs.

Thanks for all the great contributors on GitHub!🔥⚡🔥

## Contents

- [A Survey on Data-Centric LLMs Lifecycle](#a-survey-on-data-centric-llms-lifecycle)
  - [Contents](#contents)
  - [1. Surveys](#1-surveys)
  - [2. Taxonomy](#2-taxonomy)
  - [3. Existing Datasets](#3-existing-datasets)
    - 3.1 General Domain
      - [3.1.1 Pretrain](#311-pretrain)
      - [3.1.2 SFT](#312-sft)
      - [3.1.3 RL](#313-rl)
    - 3.2 Specific Domain
      - [3.2.1 Reasoning and Code](#321-reasoning-and-code)
      - [3.2.2 Safety and Alignment](#322-safety-and-alignment)
      - [3.2.3 Agent and Tool Use](#323-agent-and-tool-use)
    - [3.3 Summary](#33-summary)
  - [4. Creation](#4-creation)
      - 4.1 Annotation
        - [4.1.1 Data Processing](#411-data-processing)      
        - [4.1.2 Prompt Engineering](#412-prompt-engineering)
      - 4.2 Synthesis
        - [4.2.1 Sampling-Based](#421-sampling-based)
        - [4.2.2 Data Transformation](#422-data-transformation)
        - [4.2.3 Back-Translation](#423-back-translation)
        - [4.2.4 Human-AI Collaboration](#424-human-ai-collaboration)
        - [4.2.5 Symbolic Generation](#425-symbolic-generation)
      - 4.3 Selection
        - [4.3.1 Diversity](#431-diversity)
        - [4.3.2 Quality](#432-quality)
        - [4.3.3 Composite Strategy](#433-composite-strategy)
      - [4.4 Summary](#44-summary)
  - [5. Strategy](#5-strategy)
    - [5.1 Effectiveness](#51-effectiveness)
      - [5.1.1 Sample Construction](#511-sample-construction)
      - [5.1.2 Sample Mixing](#512-sample-mixing)
    - [5.2 Efficiency](#52-efficiency)
      - [5.2.1 Mid Training](#521-mid-training)
      - [5.2.2 Multi-Stage Training](#522-multi-stage-training)
    - [5.3 Integration](#53-integration)
      - [5.3.1 Constrain](#531-constrain)
      - [5.3.2 Cognitive Alignment](#532-cognitive-alignment)
    - [5.4 Truthfulness](#54-truthfulness)
      - [5.4.1 Consistency](#541-consistency)
      - [5.4.2 RAG](#542-rag)
      - [5.4.3 Decoding](#543-decoding)
  - [6. Future Trends]
    - [6.1 Dataset]
      - [6.1.1 Domain-Specific Data Scarcity](#611-domain-specific-data-scarcity)
        - [Craft Your Dataset: Task-Specific Synthetic Dataset Generation Through Corpus Retrieval and Augmentation](https://arxiv.org/abs/2404.07991). TACL 2025.

      - [6.1.2 Cross-Domain Data Reutilization](#612-cross-domain-data-reutilization)
        - [Synthetic Continued Pretraining](https://arxiv.org/abs/2409.07459). ICLR 2025.

      - [6.1.3 Data Sensitivity and Privacy](#613-data-sensitivity-and-privacy)
        - [Differentially Private Synthetic Data via Foundation Model APIs 2: Text](https://arxiv.org/abs/2403.01749). ICML 2024.
    - [6.2 Creation]
      - [6.2.1 Data Agent Pipelines](#621-data-agent-pipelines)
        - [DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows](https://aclanthology.org/2024.acl-long.208/). ACL 2024.

      - [6.2.2 Multi-source Synthesis](#622-multi-source-synthesis)
        - [Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement](https://aclanthology.org/2025.acl-long.1091/). ACL 2025.

      - [6.2.3 Data Flywheel](#623-data-flywheel)
        - [Recycle-The-Web: A Method to Enhance Pre-training Data Quality and Quantity for Language Models](https://arxiv.org/abs/2408.09812). ACL 2025 Findings.

      - [6.2.4 User Collaboration](#624-user-collaboration)
        - [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2401.12968). ICLR 2025.

      - [6.2.5 Cross-modal Knowledge Utilization](#625-cross-modal-knowledge-utilization)
        - [TextHarmony: A Unified Architecture for Multimodal LLMs Integrating Vision, Language and Diffusion](https://arxiv.org/abs/2407.12773). NeurIPS 2024.

      - [6.2.6 Data Efficiency](#626-data-efficiency)
        - [Less: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/abs/2402.04333). ICML 2024.

      - [6.2.7 Length Control](#627-length-control)
        - [Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/abs/2310.18558). TMLR 2025.

      - [6.2.8 Model-Aware Filter](#628-model-aware-filter)
        - [SeaPO: Strategic Error Amplification for Robust Preference Optimization](https://arxiv.org/abs/2409.12512). EMNLP 2025 Findings.
    - [6.3 Strategy]
      - [6.3.1 Unified Post-Training](#631-unified-post-training)
        - [SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning](https://arxiv.org/abs/2506.19767). arXiv 2025.

      - [6.3.2 Cross-Stage Data Reutilization](#632-cross-stage-data-reutilization)
        - [Physics of Language Models: Part 3.1, Knowledge Storage and Extraction](https://arxiv.org/abs/2310.06889). arXiv 2024.

      - [6.3.3 Test-Time Strategy](#633-test-time-strategy)
        - [CarBoN: Calibrated Best-of-N Sampling Improves Test-Time Reasoning](https://arxiv.org/abs/2505.15707). ACL 2025.

      - [6.3.4 Data Unlearning](#634-data-unlearning)
        - [Large Language Model Unlearning](https://arxiv.org/abs/2310.10058). NeurIPS 2024.

## 1. Surveys

### Data Selection
- A survey on data selection for language models
- A survey on Data selection for llm instruction tuning
- The art of data selection: A survey on Data Selection for Fine-tuning large language models
### Generation
- [**Best Practices and Lessons Learned on Synthetic Data for Language Models**](https://arxiv.org/abs/2404.07503) *Ruibo Liu, Jerry Wei, Fangyu Liu, Chenglei Si, Yanzhe Zhang, Jinmeng Rao, Steven Zheng, Daiyi Peng, Diyi Yang, Denny Zhou, Andrew M. Dai.* COLM 2024.
- [**On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey**](https://arxiv.org/abs/2406.15126) *Lin Long, Rui Wang, Ruixuan Xiao, Junbo Zhao, Xiao Ding, Gang Chen, Haobo Wang.* Arxiv 2024.
- [**Large Language Models for Data Annotation: A Survey**](https://arxiv.org/abs/2402.13446) *Zhen Tan, Dawei Li, Song Wang, Alimohammad Beigi, Bohan Jiang, Amrita Bhattacharjee, Mansooreh Karami, Jundong Li, Lu Cheng, Huan Liu.* Arxiv 2024.
- A Survey of LLM × DATA
- Knowledge Distillation and Dataset Distillation of Large Language Models: Emerging Trends, Challenges, and Future Directions
- A Survey on Data Synthesis and Augmentation for Large Language Models
- Data-centric Artificial Intelligence: A Survey
- Automatically Correcting Large Language Models : Surveying the Landscape of Diverse Automated Correction Strategies
- Survey on Knowledge Distillation for Large Language Models: Methods, Evaluation, and Application
- Data Augmentation using LLMs: Data Perspectives, Learning Paradigms and Challenges
- An Empirical Survey of Data Augmentation for Limited Data Learning in NLP
- AI Alignment: A Comprehensive Survey
- Quality, Diversity, and Complexity in Synthetic Data SURVEYING THE EFFECTS OF QUALITY, DIVERSITY, AND COMPLEXITY IN SYNTHETIC DATA FROM LARGE LANGUAGE MODELS
- A Survey of Multimodal Large Language Model from A Data-centric Perspective
- A Survey of Post-Training Scaling in Large Language Models
- Rethinking Data Mixture for Large Language Models: A Comprehensive Survey and New Perspectives
selection
### Blogs

- [**Synthetic dataset generation techniques: Self-Instruct**](https://huggingface.co/blog/davanstrien/self-instruct) *Daniel van Strien.* 2024
- [**LLM-Driven Synthetic Data Generation, Curation & Evaluation**](https://cobusgreyling.medium.com/llm-driven-synthetic-data-generation-curation-evaluation-33731e33b525) *Cobus Greyling.* 2024
- [**The Rise of Agentic Data Generation**](https://huggingface.co/blog/mlabonne/agentic-datagen) *Maxime Labonne.* 2024
- https://blog.csdn.net/qq_43688587/article/details/148533722 
## 2. Taxonomy

## 3. Existing Datasets
### 3.1 General Domain
#### 3.1.1 Pretrain
- [SlimPajama: A 627B token cleaned and deduplicated version of RedPajama](https://arxiv.org/abs/2305.09593). NeurIPS 2023 Datasets & Benchmarks.
- [Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://aclanthology.org/2024.acl-long.840/). ACL 2024.
- [DCLM: Data-Centric Language Modeling](https://arxiv.org/abs/2405.07424). arXiv 2024.
- [GneissWeb: Preparing High Quality Data for LLMs at Scale](https://arxiv.org/abs/2502.14907). arXiv 2025.
#### 3.1.2 SFT
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416). JMLR 2024.
- [Dolly: Free Dolly—Introducing the World's First Truly Open Instruction-Tuned LLM](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm). Databricks Blog 2023.
- [Alpaca: A Strong, Replicable Instruction-Following Model](https://github.com/tatsu-lab/stanford_alpaca). Stanford 2023.
- [Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2306.02707). ICML 2023.
- [Tülu: How Far Can Camels Go?](https://arxiv.org/abs/2306.04751). NeurIPS 2023.
- [Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data](https://aclanthology.org/2023.emnlp-main.385/). EMNLP 2023.
#### 3.1.3 RL
- [InstructGPT: Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155). NeurIPS 2022.
- [RLAIF: Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073). NeurIPS 2022.
- [UltraFeedback: Boosting Language Models with High-Quality Feedback](https://arxiv.org/abs/2310.01377). ICML 2024.
- [Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124). arXiv 2024.
- [ORM/PRM: Let's Verify Step by Step](https://arxiv.org/abs/2305.20050). ICLR 2024.

  
### 3.2 Specific Domain
#### 3.2.1 Reasoning and Code
- [GSM8K: Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168). NeurIPS 2021.
- [MATH: Measuring Mathematical Problem Solving with the MATH Dataset](https://arxiv.org/abs/2103.03874). NeurIPS 2021 Datasets & Benchmarks.
- [OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems](https://aclanthology.org/2024.acl-long.211/). ACL 2024.
- [MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models](https://arxiv.org/abs/2309.12284). ICLR 2024.
- [MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning](https://arxiv.org/abs/2309.05653). ICLR 2024.
- [HumanEval: Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374). ICML 2021.
- [MBPP: Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732). arXiv 2021.
- [MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation](https://arxiv.org/abs/2208.08227). ACL 2022.
- [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](https://arxiv.org/abs/2306.08568). ICLR 2024.
- [Magicoder: Empowering Code Generation with OSS-Instruct](https://arxiv.org/abs/2312.02120). ICML 2024.

#### 3.2.2 Safety and Alignment
- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://aclanthology.org/2022.acl-long.229/). ACL 2022.
- [AlpacaEval: An Automatic Evaluator of Instruction-Following Models](https://arxiv.org/abs/2305.14387). arXiv 2023.
- [MT-Bench: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685). NeurIPS 2023.
- [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://aclanthology.org/2024.naacl-long.301/). NAACL 2024.
- [HarmBench: A Standardized Evaluation Framework for Automated Red Teaming](https://arxiv.org/abs/2402.11766). ICML 2024.
#### 3.2.3 Agent and Tool Use
- [API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs](https://aclanthology.org/2023.emnlp-main.187/). EMNLP 2023.
- [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789). ICLR 2024.
- [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688). ICLR 2024.
- [GAIA: A Benchmark for General AI Assistants](https://arxiv.org/abs/2312.13130). ICLR 2024.
- [ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities](https://aclanthology.org/2025.findings-naacl.65/). NAACL 2025 Findings.
- [BFCL: The Berkeley Function Calling Leaderboard](https://arxiv.org/abs/2407.00135). ICML 2025.
- [APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://arxiv.org/abs/2406.18518). NeurIPS 2024 Datasets & Benchmarks.
- [AgentInstruct: Toward Generative Teaching with Agentic Flows](https://arxiv.org/abs/2407.03502). arXiv 2024.
- [Magnet: Multi-Turn Tool-Use Data Synthesis and Distillation via Graph Translation](https://aclanthology.org/2025.acl-long.1566/). ACL 2025.
- [T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning](https://arxiv.org/abs/2505.16986). ACL 2025.
- [The Behavior Gap: Evaluating Zero-Shot LLM Agents in Complex Task-Oriented Dialogs](https://aclanthology.org/2025.findings-acl.1205/). ACL 2025 Findings.

### 3.3 Summary
### 4. Creation
##### 4.1.1 Data Processing
- [Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality LLM Training Data](https://arxiv.org/abs/2505.05427). arXiv 2025.
- [Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning](https://arxiv.org/abs/2505.18347). ICML 2025.
- [AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark](https://arxiv.org/abs/2506.04411). EMNLP 2025 Findings.
##### 4.1.2 Prompt Engineering
- [Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding](https://arxiv.org/abs/2401.04398). ICLR 2024.
- [Task Decomposition Improves Understanding in Large Language Models](https://aclanthology.org/2024.naacl-long.106/). NAACL 2024.
- [PE2: Prompt Engineering a Prompt Engineer](https://aclanthology.org/2024.findings-acl.21/). ACL 2024 Findings.
- [OPRO: Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409). ICLR 2024.
### 4.2 Synthesis
#### 4.2.1 Sampling-Based
- [Bonito: Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation](https://arxiv.org/abs/2402.11107). ACL 2024 Findings.
- [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2401.12968). ICLR 2025.
- [AQuilt: Weaving Logic and Self-Inspection into Low-Cost, High-Relevance Data Synthesis for Specialist LLMs](https://arxiv.org/abs/2409.15617). EMNLP 2025.
#### 4.2.2 Data Transformation
- [Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement](https://aclanthology.org/2025.acl-long.1091/). ACL 2025.
- [Tree-KG: An Expandable Knowledge Graph Construction Framework for Knowledge-Intensive Domains](https://aclanthology.org/2025.acl-long.907/). ACL 2025.
- [Personas: Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/abs/2406.20094). arXiv 2024.
- [Reformatted Alignment](https://aclanthology.org/2024.findings-emnlp.32/). EMNLP 2024 Findings.
#### 4.2.3 Back-Translation
- [Back-and-orth Translation: Better Alignment with Instruction Back-and-orth Translation](https://aclanthology.org/2024.findings-emnlp.777/). EMNLP 2024 Findings.
- [Safer-Instruct: Aligning Language Models with Automated Preference Data](https://aclanthology.org/2024.naacl-long.422/). NAACL 2024.
- [Cycle-Instruct: Fully Seed-Free Instruction Tuning via Dual Self-Training and Cycle Consistency](https://arxiv.org/abs/2505.19139). ACL 2025.

#### 4.2.4 Human-AI Collaboration
- [SALMON: Self-Alignment with Instructable Reward Models](https://arxiv.org/abs/2310.14986). ICLR 2024.
- [AutoIF: Self-Play with Execution Feedback Improves Instruction-Following](https://arxiv.org/abs/2405.12995). ICLR 2025.
- [APT: Improving Specialist LLM Performance with Weakness Case Acquisition and Iterative Preference Training](https://aclanthology.org/2025.findings-acl.1079/). ACL 2025 Findings.
- [SwS: Self-Aware Weakness-Driven Problem Synthesis in Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2503.21782). ICLR 2025.
- [Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning](https://aclanthology.org/2025.acl-long.417/). ACL 2025.
- [LEMMA: Learning from Errors for Mathematical Advancement in LLMs](https://aclanthology.org/2025.findings-acl.605/). ACL 2025 Findings.
- [RISE: Subtle Errors in Reasoning — Preference Learning via Error-Injected Self-Editing](https://aclanthology.org/2025.acl-long.1506/). ACL 2025.
#### 4.2.5 Symbolic Generation
- [SR-LCL: Symbolic Regression with a Learned Concept Library](https://proceedings.neurips.cc/paper_files/paper/2024/file/4ec3ddc465c6d650c9c419fb91f1c00a-Paper-Conference.pdf). NeurIPS 2024.
- [SymbCoT: Faithful Logical Reasoning via Symbolic Chain-of-Thought](https://aclanthology.org/2024.acl-long.720/). ACL 2024.
- [ProtoReasoning: Prototypes as the Foundation for Generalizable Reasoning in LLMs](https://arxiv.org/abs/2503.19556). arXiv 2025.
- [Aristotle: Mastering Logical Reasoning with a Logic-Complete Decompose-Search-Resolve Framework](https://aclanthology.org/2025.acl-long.153/). ACL 2025.
- [Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models](https://aclanthology.org/2025.acl-long.785/). ACL 2025.
- [SC-NeuroSymbolic: Sound and Complete Neurosymbolic Reasoning with LLM-Grounded Interpretations](https://arxiv.org/abs/2406.01593). ICLR 2025.
- [SymbolicThought: Integrating Language Models and Symbolic Reasoning for Consistent and Interpretable Human Relationship Understanding](https://arxiv.org/abs/2502.13756). arXiv 2025.
- [MURI: High-Quality Instruction Tuning Datasets for Low-Resource Languages via Reverse Instructions](https://aclanthology.org/2024.findings-emnlp.414/). EMNLP 2024 Findings.
- [AdaCoT: Adaptive Multilingual Chain-of-Thought for Cross-Lingual Factual Reasoning](https://arxiv.org/abs/2502.13756). ACL 2025 Findings.
### 4.3 Selection
#### 4.3.1 Diversity
- [Impossible Distillation: Gradient-Based Data Diversification Boosts Generalization in LLM Reasoning](https://arxiv.org/abs/2402.13669). ACL 2024.
- [QDIT: Data Diversity Matters for Robust Instruction Tuning](https://aclanthology.org/2024.findings-emnlp.195/). EMNLP 2024 Findings.
- [Instag: Instruction Tagging for Analyzing Supervised Fine-Tuning of Large Language Models](https://arxiv.org/abs/2308.07074). ICLR 2024.
- [G-Vendi: Data Selection with Model-Based Diversity](https://arxiv.org/abs/2503.21744). ICLR 2025.

#### 4.3.2 Quality
- [LIFT: Rethinking the Instruction Quality — LIFT Is What You Need](https://arxiv.org/abs/2310.11691). EMNLP 2023 Findings.
- [Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models](https://aclanthology.org/2024.emnlp-main.248/). EMNLP 2024.
- [IFD: Instruction-Following Difficulty Score for Data Selection](https://arxiv.org/abs/2403.02980). ICLR 2024.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948). arXiv 2025.
#### 4.3.3 Composite Strategy
- [SelectIT: Selective Instruction Tuning for LLMs via Uncertainty-Aware Self-Reflection](https://arxiv.org/abs/2403.07392). NeurIPS 2024.
- [Dataman: Data Manager for Pre-Training Large Language Models](https://arxiv.org/abs/2501.00656). ICLR 2025.
### 4.4 Summary

## 5. Strategy
### 5.1 Effectiveness
#### 5.1.1 Sample Construction
- [CommonIT: Commonality-Aware Instruction Tuning for Large Language Models via Data Partitions](https://aclanthology.org/2024.emnlp-main.561/). EMNLP 2024.
- [Persona-Plug: LLMs + Persona-Plug = Personalized LLMs](https://aclanthology.org/2025.acl-long.461/). ACL 2025.
#### 5.1.2 Sample Mixing
- [D4: Improving LLM Pretraining via Document De-Duplication and Diversification](https://arxiv.org/abs/2305.10429). NeurIPS 2023 Datasets & Benchmarks.
- [QuaDMix: Quality-Diversity Balanced Data Selection for Efficient LLM Pretraining](https://arxiv.org/abs/2502.13756). ICLR 2025.
- [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186). arXiv 2024.
- [Baichuan-M2: Scaling Medical Capability with Large Verifier System](https://arxiv.org/abs/2501.12236). arXiv 2025.

### 5.2 Efficiency
#### 5.2.1 Mid Training
- [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471). arXiv 2025.
- [OctoThinker: Mid-Training Incentivizes Reinforcement Learning Scaling](https://arxiv.org/abs/2505.15707). ICLR 2025.
- [RA3: Sequential Variational Lower Bound Optimization via RL](https://arxiv.org/abs/2502.14799). ICLR 2025.

#### 5.2.2 Multi-Stage Training
- [InsCL: A Data-Efficient Continual Learning Paradigm for Fine-Tuning Large Language Models with Instructions](https://aclanthology.org/2024.naacl-long.37/). NAACL 2024.
- [DMT: Dynamic Multi-Task Training for Large Language Models](https://arxiv.org/abs/2403.07816). ICLR 2024.
- [SDFT: Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning](https://aclanthology.org/2024.acl-long.58/). ACL 2024.
- [Exploring Forgetting in Large Language Model Pre-Training](https://aclanthology.org/2025.acl-long.105/). ACL 2025.
  
### 5.3 Integration
#### 5.3.1 Constrain
- [Function Vectors: Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting](https://arxiv.org/abs/2409.17400). ICLR 2025.
- [Spurious Forgetting in Continual Learning of Language Models](https://arxiv.org/abs/2406.05224). ICLR 2025.

#### 5.3.2 Cognitive Alignment
- [Cognitive Behaviors That Enable Self-Improving Reasoners](https://openreview.net/forum?id=QGJ9ttXLTy). CoLM 2025.

### 5.4 Truthfulness
#### 5.4.1 Consistency
- [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.09098). ICML 2025.
- [MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration](https://arxiv.org/abs/2406.08490). ACL 2025.
- [Self-Certainty: Scalable Best-of-N Selection for Large Language Models via Self-Certainty](https://arxiv.org/abs/2502.14756). ICLR 2025.

#### 5.4.2 RAG
- [SEER: Self-Aligned Evidence Extraction for Retrieval-Augmented Generation](https://aclanthology.org/2024.emnlp-main.178/). EMNLP 2024.
- [Rowen: Adaptive Retrieval-Augmented Generation for Hallucination Mitigation in LLMs](https://arxiv.org/abs/2504.09959). ACL 2025.
#### 5.4.3 Decoding
- [ROSE: Reverse Prompt Contrastive Decoding Boosts Safety of Instruction-Tuned LLMs](https://aclanthology.org/2024.findings-acl.814/). ACL 2024 Findings.

### 5.5 Summary

