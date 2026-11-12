# Paper Quality Ranking

> Auto-generated from `data/papers.json` by `scripts/rank_papers.py` using [`config/quality_ranking.yaml`](config/quality_ranking.yaml).
> Last refreshed: **2026-11-12 03:01 UTC** · 2,807 papers.

**This is a reproducible proxy score, not a judgement of scientific quality.** The corpus has no citation counts, review decisions, or full texts; every point comes from a boolean metadata signal or a bounded normalisation. Use it to prioritise reading and curation, then apply human judgement.

## The rule

`score = evidence + reproducibility + maturity + fit + clarity − penalty`, clipped to 0–100.

| Dimension | Max | Signals (points) |
| --- | --- | --- |
| evidence · 实证强度 | 30 | `numeric_result` +10, `sota_claim` +6, `experiments` +6, `baseline` +5, `ablation` +3 |
| reproducibility · 可复现性 | 25 | `code_link` +10, `release_claim` +8, `artifact` +7 |
| maturity · 文稿成熟度 | 15 | `peer_reviewed` +8, `revised` +4, `multi_author` +3 |
| fit · 主题契合与广度 | 20 | `relevance_norm` +12, `families` +4, `cross_tier` +2, `categories` +2 |
| clarity · 贡献明确度 | 10 | `contribution_sentence` +4, `named_method` +3, `abstract_length` +3 |

Penalty: subcategory `Other` with relevance < 8 gets **-5** (classification-boundary hits).
Grades: **A** ≥ 55 · **B** ≥ 45 · **C** ≥ 32 · **D** below that.

Signal definitions are regexes in the config file; `relevance_norm` is `min(relevance_score, cap) / cap × 12`. `quality_evidence` in `data/papers.json` records exactly which signals fired for each paper.

## Distribution

| Grade | Threshold | Papers | Share |
| --- | --- | --- | --- |
| A | 55 | 182 | 6.5% |
| B | 45 | 469 | 16.7% |
| C | 32 | 1,191 | 42.4% |
| D | — | 965 | 34.4% |

Score quantiles: p10 **23.8** · p50 **36.2** · p90 **52.2** · max **76.6**.

| Primary tier | Papers | Mean quality |
| --- | --- | --- |
| Data Ingestion Strategies | 1,518 | 38.2 |
| Data Substrates | 1,107 | 35.1 |
| Data Creation and Selection | 182 | 38.2 |

## Top 100

| # | Score | Grade | Paper | Tier / subcategory | Key evidence |
| --- | --- | --- | --- | --- | --- |
| 1 | 76.6 | A | [VCoT-SQL: Enhancing Text-to-SQL Generation through Dual Chain-of-Thought Validation Training](http://arxiv.org/abs/2611.00815v2) | Data Creation and Selection / Annotation / Processing | peer_reviewed, code_link, release_claim |
| 2 | 72.4 | A | [ReScraper: Unified Scraping and Cleaning of Web Data for Effective LLM Pretraining](http://arxiv.org/abs/2609.34287v1) | Data Substrates / Pretraining / General Data | code_link, release_claim, numeric_result |
| 3 | 71.0 | A | [SpatialChain: A Benchmark for Auditing Spatial Reasoning Faithfulness in VLMs](http://arxiv.org/abs/2610.06413v1) | Data Ingestion Strategies / Curriculum / Mid-Training / Replay | code_link, release_claim, numeric_result |
| 4 | 70.0 | A | [Behavior2Trip: Towards Personalized Travel Planning via User Behavior Trajectory](http://arxiv.org/abs/2608.26807v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, baseline |
| 5 | 69.8 | A | [GTA-RAG: Graph-Trajectory-Augmented Reinforcement Learning for Multi-Turn Retrieval-Augmented Reasoning](http://arxiv.org/abs/2608.22479v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, baseline |
| 6 | 69.8 | A | [Hybrid Reasoning Agents](http://arxiv.org/abs/2610.18861v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 7 | 68.6 | A | [Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding](http://arxiv.org/abs/2609.12243v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 8 | 68.2 | A | [FastBench: Can Streaming VLMs Perceive High-Dynamic Real-World Streams?](http://arxiv.org/abs/2610.12427v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, numeric_result, baseline |
| 9 | 67.8 | A | [Drive the Thoughts: Runtime Monitoring of VLA Reasoning-Trajectory Consistency](http://arxiv.org/abs/2608.29583v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 10 | 67.8 | A | [Where Root Cause Analysis Fails: A Retrieval-Reranking Decomposition](http://arxiv.org/abs/2609.36686v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 11 | 67.6 | A | [RES-MR: Risk-Aware Reasoning for Explainable and Safe Medication Recommendation](http://arxiv.org/abs/2610.14474v1) | Data Substrates / Other | code_link, release_claim, baseline |
| 12 | 67.4 | A | [Knowledge Graphs as Files: An Agent Harness Where Tool Creation and Tool Use Co-Evolve](http://arxiv.org/abs/2610.18616v2) | Data Creation and Selection / Transformation | code_link, release_claim, numeric_result |
| 13 | 67.4 | A | [Rollout-Time Intervention: Unlocking Agentic Reinforcement Learning Beyond Policy Limits](http://arxiv.org/abs/2610.13438v2) | Data Substrates / Other | code_link, release_claim, baseline |
| 14 | 66.6 | A | [SEAD: A State-Based Perspective on Attack and Defense in Tool-Using Agents](http://arxiv.org/abs/2609.34518v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 15 | 66.4 | A | [Beyond Factual Knowledge: Benchmarking and Learning Step-Level Procedural Rule Reasoning in Large Language Models](http://arxiv.org/abs/2608.22753v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, numeric_result, baseline |
| 16 | 66.4 | A | [TALK-Dem: Benchmarking Embodied Task Planning under Dementia-Associated Communication Patterns](http://arxiv.org/abs/2609.38371v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, numeric_result, baseline |
| 17 | 66.2 | A | [Seek: Self-Evaluative Exploration for Knowledge Retrieval](http://arxiv.org/abs/2609.28980v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | peer_reviewed, numeric_result, baseline |
| 18 | 65.8 | A | [Test-time Calibration Learning for Large Language Model Reasoning](http://arxiv.org/abs/2610.02695v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 19 | 65.8 | A | [RRSI: Regularized Recursive Self-Improvement of Agent Harnesses](http://arxiv.org/abs/2609.24972v2) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, ablation |
| 20 | 65.8 | A | [SEER: Long-Context Reasoning via Selective Visual-Text Compression](http://arxiv.org/abs/2608.15962v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, numeric_result, baseline |
| 21 | 65.6 | A | [ContextPilot: Teaching Agents for Proactive Context Management via Fine-grained RL](http://arxiv.org/abs/2608.28476v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, baseline |
| 22 | 65.6 | A | [LongGuard: Mechanistic Analysis and Training-Free Mitigation of Long-Context Failure in Safety Guardrails](http://arxiv.org/abs/2608.27580v2) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | release_claim, numeric_result, baseline |
| 23 | 65.6 | A | [MemGuard: Persisting Verifier Signals for LLM-Agent Memory Governance](http://arxiv.org/abs/2608.21867v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 24 | 65.2 | A | [Route Me If You Can: A Benchmark for Query Reformulation Selection](http://arxiv.org/abs/2609.14885v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | peer_reviewed, code_link, release_claim |
| 25 | 65.0 | A | [Feeding BabyLMs Macaroni: Code-Switching Curricula Cause Cross-Lingual Convergence](http://arxiv.org/abs/2609.30535v1) | Data Ingestion Strategies / Curriculum / Mid-Training / Replay | code_link, release_claim, baseline |
| 26 | 64.8 | A | [Adaptive Critical Token-Aware Retrieval for Repository-Level Code Generation](http://arxiv.org/abs/2609.01601v1) | Data Substrates / Reasoning and Code | code_link, numeric_result |
| 27 | 64.4 | A | [How Robust Are Automated Fact-Checking Systems? A Cross-Benchmark Evaluation](http://arxiv.org/abs/2608.25934v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | release_claim, numeric_result, baseline |
| 28 | 64.4 | A | [OptiSkill: A Hierarchical and Evolving SkillBank for LLM-Based Optimization Modeling](http://arxiv.org/abs/2609.22987v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, baseline |
| 29 | 64.4 | A | [Verify Smarter, Evolve Further: Efficient Harness Evolution through Behavior-Aware Verification](http://arxiv.org/abs/2608.27311v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 30 | 64.2 | A | [Safe, Persistent, and Evolving Agent Harness for Understanding Partially Observable Worlds](http://arxiv.org/abs/2610.11552v1) | Data Substrates / Other | code_link, release_claim, numeric_result |
| 31 | 64.0 | A | [EDGE: Experience-Distillation for Guided Exploration in Agentic Reinforcement Learning](http://arxiv.org/abs/2608.21946v2) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 32 | 64.0 | A | [OntologyAligner: Ontology-Aligned Retrieval and Hierarchy-Guided Large Language Model Reranking for Biomedical Ontology Normalization](http://arxiv.org/abs/2609.10055v1) | Data Substrates / Other | code_link, ablation, numeric_result |
| 33 | 64.0 | A | [sk-bench: A Native-First Benchmark for Evaluating Large Language Models in Slovak](http://arxiv.org/abs/2610.09152v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 34 | 63.8 | A | [Colorectal Cancer Segmentation with Adaptive Augmentation and Multi-Resolution Ensemble Models](http://arxiv.org/abs/2609.38419v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | peer_reviewed, code_link, release_claim |
| 35 | 63.8 | A | [EnSI-RAG: Entity-Structure-Indexed Retrieval-Augmented Generation for Long-Document Question Answering](http://arxiv.org/abs/2608.21252v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 36 | 63.8 | A | [MUSE: Multi-Level Memory with Selective Runtime Injection for Scientific Coding Agents](http://arxiv.org/abs/2610.38332v1) | Data Substrates / Other | code_link, release_claim, baseline |
| 37 | 63.8 | A | [Principia: Test-Time Hypotheses to Principle Construction for Scientific Discovery](http://arxiv.org/abs/2610.32195v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, baseline |
| 38 | 63.6 | A | [SAGE: Semantic Anchor-Guided Evolution for Grounded Medical QA Data Synthesis](http://arxiv.org/abs/2610.08093v1) | Data Creation and Selection / Synthetic Data | code_link, release_claim |
| 39 | 63.4 | A | [Dependency-Aware Trajectory Refinement for Efficient Multi-Turn Agent Fine-Tuning](http://arxiv.org/abs/2609.18417v1) | Data Substrates / Agent and Tool Use | release_claim, numeric_result, baseline |
| 40 | 63.4 | A | [Compute Has Shape: Self-Morphing the Internal Computation of Frozen LLMs](http://arxiv.org/abs/2610.38382v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 41 | 63.4 | A | [EviCast: Forecasting Retrieval Gains Through Shared Response Calibration](http://arxiv.org/abs/2611.02387v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 42 | 63.2 | A | [Inducing Process Supervision from Outcome-Only Reinforcement Learning](http://arxiv.org/abs/2609.36641v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim |
| 43 | 63.2 | A | [ISO-RAG: Isoperimetric Noise Control for Retrieval-Augmented Generation](http://arxiv.org/abs/2609.00513v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, baseline |
| 44 | 63.2 | A | [SIC-Agents: Benchmarking and Building an Adaptive Simulator for Pediatric Serious Illness Communication Training](http://arxiv.org/abs/2608.29481v1) | Data Ingestion Strategies / Curriculum / Mid-Training / Replay | code_link, release_claim, baseline |
| 45 | 63.0 | A | [MABPD: Multi-Agent Bias Probing & Detection via Structured Argument Debate](http://arxiv.org/abs/2609.04841v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | release_claim, ablation, numeric_result |
| 46 | 62.8 | A | [MERGE: Multi-LLM Ensemble for Retrieval via Generative Enrichment](http://arxiv.org/abs/2609.37574v2) | Data Substrates / Other | numeric_result, baseline, revised |
| 47 | 62.6 | A | [Foresight-over-Graph: Reasoning Beyond Local Horizons for Knowledge Base Question Answering](http://arxiv.org/abs/2610.08388v1) | Data Substrates / Other | code_link, release_claim, numeric_result |
| 48 | 62.6 | A | [One Policy, Any Budget: Internalizing Budget-Aware Search via Reinforcement Learning](http://arxiv.org/abs/2609.00813v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, baseline |
| 49 | 62.6 | A | [Towards Efficient Reasoning in LLM-Based Recommender Systems via Model Merging](http://arxiv.org/abs/2608.10447v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, baseline |
| 50 | 62.4 | A | [HGP:An on-device personalized agent memory via hybrid graph storage](http://arxiv.org/abs/2610.10071v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, numeric_result, baseline |
| 51 | 62.2 | A | [Axis-LoRA: Probing and Adapting Embedding Models for Personal-Memory Retrieval](http://arxiv.org/abs/2611.07820v1) | Data Creation and Selection / Transformation | code_link, release_claim, numeric_result |
| 52 | 62.2 | A | [Large Language Models Systematically Favor Popular Options: Evidence and Mitigation Across MCQs](http://arxiv.org/abs/2608.29257v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 53 | 62.0 | A | [DocTalkBN: A Novel Dataset of Expert Telemedicine Conversations in Bengali](http://arxiv.org/abs/2608.27110v1) | Data Creation and Selection / Synthetic Data | code_link, release_claim, baseline |
| 54 | 62.0 | A | [A Closer Look at Agentic BBO: Benchmarking LLM Agents for Black-Box Optimization](http://arxiv.org/abs/2610.12183v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim |
| 55 | 62.0 | A | [ScopeIF: Improving Scope-Aware Precise Instruction-Following in Large Language Models via Graded Reward Modeling](http://arxiv.org/abs/2609.32189v1) | Data Substrates / Instruction / Preference Data | code_link, release_claim |
| 56 | 61.8 | A | [EvoRouter: Cost-Efficient Multi-Agent Routing via Insight-Guided On-Policy Distillation](http://arxiv.org/abs/2610.35690v1) | Data Substrates / Other | code_link, numeric_result, baseline |
| 57 | 61.6 | A | [TriCalRAG: A Three-Strategy, Retrieval-Augmented Benchmark for On-Premise LLM-Based Root Cause Analysis in AIOps](http://arxiv.org/abs/2609.14762v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | release_claim, numeric_result, baseline |
| 58 | 61.0 | A | [AttnSR: Attention Step Reward for Long-Horizon Agentic RL](http://arxiv.org/abs/2611.07932v1) | Data Ingestion Strategies / Curriculum / Mid-Training / Replay | code_link, release_claim, ablation |
| 59 | 61.0 | A | [CityPlanner: A Sandbox Agent for Executable Urban Planning](http://arxiv.org/abs/2609.09578v1) | Data Substrates / Other | code_link, release_claim, baseline |
| 60 | 61.0 | A | [SemPOI-RL: Aligning LLM Semantic Reasoning for Interpretable Out-of-Town POI Sequential Generation](http://arxiv.org/abs/2608.30399v1) | Data Substrates / Other | code_link, release_claim, baseline |
| 61 | 60.6 | A | [SWE-Critix: Post-Training Generalist Verifiers for Assessing Coding-Agent Trajectories](http://arxiv.org/abs/2610.13356v2) | Data Substrates / Agent and Tool Use | release_claim, numeric_result, revised |
| 62 | 60.6 | A | [TGRL: Temperature-Grouped Reinforcement Learning for Efficient Exploration in LLMs](http://arxiv.org/abs/2609.33589v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 63 | 60.4 | A | [Act on Your Behalf, Beyond Your Authorization: Benchmarking LLM Agent Authorization Alignment](http://arxiv.org/abs/2610.27548v2) | Data Substrates / Other | code_link, release_claim, baseline |
| 64 | 60.4 | A | [Agent-Reflex: Teaching Language Agents to Act Reflectively](http://arxiv.org/abs/2610.17953v1) | Data Substrates / Agent and Tool Use | code_link, release_claim, baseline |
| 65 | 60.4 | A | [Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems](http://arxiv.org/abs/2608.21095v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, baseline |
| 66 | 60.2 | A | [Token-Disentangled Latent Test-Time Scaling for Vision-Language Reasoning](http://arxiv.org/abs/2609.35228v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, baseline |
| 67 | 60.2 | A | [A Guideline-Augmented Multi-Agent Framework for Schema-as-Code Biomedical Named Entity Recognition](http://arxiv.org/abs/2610.02970v2) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | peer_reviewed, ablation, baseline |
| 68 | 60.2 | A | [Training-Free Pseudo-Fusion for Composed Image Retrieval with Diffusion Models and Multimodal Large Language Models](http://arxiv.org/abs/2608.23102v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | peer_reviewed, code_link, baseline |
| 69 | 60.0 | A | [DataCanvas-EDU: An Agentic Framework for Instructor-Guided Synthetic Data Generation in Business Analytics Education](http://arxiv.org/abs/2609.19617v1) | Data Creation and Selection / Synthetic Data | code_link, release_claim |
| 70 | 60.0 | A | [Consolidating Reasoning with Test-Time Training](http://arxiv.org/abs/2611.02104v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | numeric_result, baseline |
| 71 | 59.8 | A | [Industrial-Instruction: An End-to-End Framework for Building Instruction-Tuning and Benchmark Datasets from Industrial Technical Reports](http://arxiv.org/abs/2608.22817v1) | Data Creation and Selection / Synthetic Data | release_claim, numeric_result, baseline |
| 72 | 59.8 | A | [Agent4RE: A Self-Refining Multi-agent Framework for End-to-End Software Requirements Engineering and Benchmarking](http://arxiv.org/abs/2610.10628v1) | Data Substrates / Other | peer_reviewed, numeric_result, baseline |
| 73 | 59.8 | A | [Skill2Query: Exploiting Skill Structure to Generate Pseudo-Queries for Agent Skill Retrieval](http://arxiv.org/abs/2608.16071v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, numeric_result, baseline |
| 74 | 59.8 | A | [Validity-Aware Jailbreak Evaluation for Large Language Models](http://arxiv.org/abs/2609.00498v1) | Data Substrates / Safety and Alignment | code_link, release_claim, baseline |
| 75 | 59.6 | A | [Learning What to Fail On: Failure-Mode Contextual Bandits for Adversarial Data Curation](http://arxiv.org/abs/2608.18681v1) | Data Creation and Selection / Data Selection | peer_reviewed, numeric_result |
| 76 | 59.6 | A | [ShutdownMod: Turning Any Trajectory into a Shutdown Evaluation](http://arxiv.org/abs/2610.35486v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim |
| 77 | 59.6 | A | [EngramRAG: Dynamic Usage-Weighted Topology and Synaptic Consolidation for Multi-Hop Agentic Memory](http://arxiv.org/abs/2609.32049v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | numeric_result, baseline |
| 78 | 59.4 | A | [AgenticRag-R1: Agentic Reinforcement Learning with Stack Memory for Multi-Step Reasoning, Retrieval and Memorizing](http://arxiv.org/abs/2608.29622v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, baseline |
| 79 | 59.4 | A | [Learning to Route in Visual Space via Multi-Step Embedding Retrieval](http://arxiv.org/abs/2609.38743v2) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | numeric_result, baseline, revised |
| 80 | 59.2 | A | [ASCRIBE: Atomic and Significance-Based Reasoning for Thai Clinical SOAP Note Generation](http://arxiv.org/abs/2610.01234v1) | Data Substrates / Pretraining / General Data | code_link, numeric_result |
| 81 | 59.2 | A | [ARCagent: An Adaptive Retrieval Calibration Agent for Clinical Question Answering](http://arxiv.org/abs/2609.36392v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, numeric_result |
| 82 | 59.2 | A | [GlucoAgent: Personalized Glucose Forecasting from Multimodal Personal Context](http://arxiv.org/abs/2611.04126v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | peer_reviewed, numeric_result, baseline |
| 83 | 59.2 | A | [LEMMO: A Large Electromagnetic Model leveraging Pretrained LLMs for Million-Sample Signal Understanding](http://arxiv.org/abs/2611.09476v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, numeric_result, baseline |
| 84 | 59.2 | A | [Think-to-Personalize: Unifying Reasoning and Retrieval for User-Centric Personalized Dense Retrieval](http://arxiv.org/abs/2608.18855v2) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | peer_reviewed, baseline, revised |
| 85 | 59.0 | A | [DEPT: Document Embedding Preservation Tuning for Unified Query Expansion and Retrieval](http://arxiv.org/abs/2608.17632v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, baseline |
| 86 | 59.0 | A | [MemCo: Memory-Centric Collaboration for Generalizing LLM Agents to Unseen Environments](http://arxiv.org/abs/2610.07376v1) | Data Substrates / Other | code_link, release_claim, baseline |
| 87 | 59.0 | A | [RSIFlow: Adaptive Self-Improvement Workflows with an Evolving Meta-Agent](http://arxiv.org/abs/2610.21791v1) | Data Substrates / Agent and Tool Use | code_link, release_claim, ablation |
| 88 | 59.0 | A | [The World as Viewed by 136 Countries: An Open, Curated, Annotated, and Multilingual Dataset of 118 Million State Media Articles](http://arxiv.org/abs/2610.24990v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim, baseline |
| 89 | 58.8 | A | [The Imitation Game: When LLMs Learn to Reason Like Programs via Code-Centric Reasoning Data Synthesis](http://arxiv.org/abs/2609.16076v1) | Data Creation and Selection / Synthetic Data | code_link, release_claim |
| 90 | 58.8 | A | [DAEDALUS: Bootstrapping Agent Memory from Self-Generated Tasks](http://arxiv.org/abs/2610.08048v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, numeric_result, baseline |
| 91 | 58.8 | A | [Toward Agentic Retrieval-Augmented Generation for Enterprise Retail Workflows over Heterogeneous Data Sources](http://arxiv.org/abs/2610.25096v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, baseline |
| 92 | 58.8 | A | [LEGO: Synergizing Expert GraphRAG and Expert Chain-of-Thought for Legal Reasoning](http://arxiv.org/abs/2609.27009v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, ablation, baseline |
| 93 | 58.8 | A | [On-policy Distillation with Verifiable Reward](http://arxiv.org/abs/2608.24696v2) | Data Substrates / Other | code_link, release_claim, revised |
| 94 | 58.8 | A | [SRPO: Self-Reflective Policy Optimization for Long-Horizon Reasoning](http://arxiv.org/abs/2608.23493v1) | Data Substrates / Other | code_link, release_claim, numeric_result |
| 95 | 58.6 | A | [VeriHarness: Scaling Agentic Verification for Long-Horizon Tasks](http://arxiv.org/abs/2610.00972v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | release_claim, numeric_result, baseline |
| 96 | 58.4 | A | [Q2D-Web: A Large-Scale Benchmark for Retrieval in Agentic RAG Systems](http://arxiv.org/abs/2609.08887v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, numeric_result |
| 97 | 58.4 | A | [MEvince: Knowledge-Constrained LLM Reasoning for Technique Identification from Malware Reports](http://arxiv.org/abs/2611.07486v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | numeric_result, baseline |
| 98 | 58.4 | A | [Well-Specified and Well-Tested? Assessing SWE-bench-Style Datasets with a Self-Evolving Agent](http://arxiv.org/abs/2610.39417v1) | Data Creation and Selection / Annotation / Processing | numeric_result, baseline |
| 99 | 58.2 | A | [AlgoWorlds: Benchmarking Tool Use for Global Optimization in Algorithmic Worlds](http://arxiv.org/abs/2608.29397v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | code_link, release_claim |
| 100 | 58.2 | A | [APEx: Distillation of Agent Procedural Experience for Adaptive Deep Research Question Answering](http://arxiv.org/abs/2609.02253v1) | Data Ingestion Strategies / Runtime Retrieval / Search / Verification | numeric_result, baseline |

## Top 15 by primary tier

### Data Substrates

| Score | Paper | Idea |
| --- | --- | --- |
| 72.4 | [ReScraper: Unified Scraping and Cleaning of Web Data for Effective LLM Pretraining](http://arxiv.org/abs/2609.34287v1) | In this work, we propose ReScraper, a unified language model of only 0.6B parameters that replaces this entire stack.… |
| 67.6 | [RES-MR: Risk-Aware Reasoning for Explainable and Safe Medication Recommendation](http://arxiv.org/abs/2610.14474v1) | To address these gaps, we propose RES-MR, a novel Risk-aware Reasoning framework for Explainable and Safe Medication Recommendation using LLMs.… |
| 67.4 | [Rollout-Time Intervention: Unlocking Agentic Reinforcement Learning Beyond Policy Limits](http://arxiv.org/abs/2610.13438v2) | We introduce Rollout-Time Intervention (RTI), a technique designed to elicit successful trajectories beyond the policy's intrinsic exploration capacit… |
| 64.8 | [Adaptive Critical Token-Aware Retrieval for Repository-Level Code Generation](http://arxiv.org/abs/2609.01601v1) | In this paper, we propose ACToR, an adaptive critical token-aware retrieval framework for repository-level code generation.… |
| 64.2 | [Safe, Persistent, and Evolving Agent Harness for Understanding Partially Observable Worlds](http://arxiv.org/abs/2610.11552v1) | To address these challenges, we introduce E-Ledger, a multi-agent harness for safe and persistent execution.… |
| 64.0 | [OntologyAligner: Ontology-Aligned Retrieval and Hierarchy-Guided Large Language Model Reranking for Biomedical Ontology Normalization](http://arxiv.org/abs/2609.10055v1) | We present OntologyAligner, a three-stage framework that combines ontology-aligned retrieval, large language model candidate reranking, and selective … |
| 63.8 | [MUSE: Multi-Level Memory with Selective Runtime Injection for Scientific Coding Agents](http://arxiv.org/abs/2610.38332v1) | We introduce MUSE, a memory management framework combining multi-level memory with selective runtime injection for scientific coding agents.… |
| 63.4 | [Dependency-Aware Trajectory Refinement for Efficient Multi-Turn Agent Fine-Tuning](http://arxiv.org/abs/2609.18417v1) | We propose viewing each trajectory as a round-level dependency DAG that exposes which rounds are globally load-bearing for the final answer, and fine-… |
| 62.8 | [MERGE: Multi-LLM Ensemble for Retrieval via Generative Enrichment](http://arxiv.org/abs/2609.37574v2) | We present MERGE (Multi-LLM Ensemble for Retrieval via Generative Enrichment), a two-stage framework: three heterogeneous 7-8B open-source LLMs indepe… |
| 62.6 | [Foresight-over-Graph: Reasoning Beyond Local Horizons for Knowledge Base Question Answering](http://arxiv.org/abs/2610.08388v1) | To address this limitation, we propose Foresight-over-Graph (FoG), a foresight-aware evidence retrieval framework for knowledge base question answerin… |
| 62.0 | [ScopeIF: Improving Scope-Aware Precise Instruction-Following in Large Language Models via Graded Reward Modeling](http://arxiv.org/abs/2609.32189v1) | To this end, we propose ScopeIF, a novel training framework for scope-aware precise instruction-following.… |
| 61.8 | [EvoRouter: Cost-Efficient Multi-Agent Routing via Insight-Guided On-Policy Distillation](http://arxiv.org/abs/2610.35690v1) | We propose EvoRouter, a self-evolving multi-agent routing framework driven by insight-guided on-policy distillation.… |
| 61.0 | [CityPlanner: A Sandbox Agent for Executable Urban Planning](http://arxiv.org/abs/2609.09578v1) | We propose CityPlanner, a sandbox-agent framework for executable urban planning.… |
| 61.0 | [SemPOI-RL: Aligning LLM Semantic Reasoning for Interpretable Out-of-Town POI Sequential Generation](http://arxiv.org/abs/2608.30399v1) | To address this gap, we propose SemPOI-RL, a framework that aligns LLM semantic reasoning with structured sequence generation for interpretable OOT re… |
| 60.6 | [SWE-Critix: Post-Training Generalist Verifiers for Assessing Coding-Agent Trajectories](http://arxiv.org/abs/2610.13356v2) | We introduce SWE-Critix, a post-training pipeline for training generalist verifiers of coding-agent trajectories.… |

### Data Creation and Selection

| Score | Paper | Idea |
| --- | --- | --- |
| 76.6 | [VCoT-SQL: Enhancing Text-to-SQL Generation through Dual Chain-of-Thought Validation Training](http://arxiv.org/abs/2611.00815v2) | We propose VCoT-SQL, a framework that improves Text-to-SQL generation through intermediate-step validated CoT supervision and a dedicated SQL validato… |
| 67.4 | [Knowledge Graphs as Files: An Agent Harness Where Tool Creation and Tool Use Co-Evolve](http://arxiv.org/abs/2610.18616v2) | We propose direct knowledge graph interaction (DKGI), a harness in which language model agents create graph access tools directly over complete node a… |
| 63.6 | [SAGE: Semantic Anchor-Guided Evolution for Grounded Medical QA Data Synthesis](http://arxiv.org/abs/2610.08093v1) | To address these obstacles, we introduce SAGE ( Semantic Anchor-Guided Evolution ), a novel data synthesis framework that enables small, locally deplo… |
| 62.2 | [Axis-LoRA: Probing and Adapting Embedding Models for Personal-Memory Retrieval](http://arxiv.org/abs/2611.07820v1) | Current personal-memory pipelines rely on frozen dense embedding models to embed memories for retrieval, but we find that these models do not capture … |
| 62.0 | [DocTalkBN: A Novel Dataset of Expert Telemedicine Conversations in Bengali](http://arxiv.org/abs/2608.27110v1) | We present DocTalkBN, a large-scale multimodal dataset of real-world expert telemedicine conversations in Bengali, collected from nationally broadcast… |
| 60.0 | [DataCanvas-EDU: An Agentic Framework for Instructor-Guided Synthetic Data Generation in Business Analytics Education](http://arxiv.org/abs/2609.19617v1) | This paper presents DataCanvas-EDU, an agentic framework for instructor-guided synthetic data generation in business analytics education.… |
| 59.8 | [Industrial-Instruction: An End-to-End Framework for Building Instruction-Tuning and Benchmark Datasets from Industrial Technical Reports](http://arxiv.org/abs/2608.22817v1) | We address this gap with Industrial-Instruction, contributing (i) two open QA datasets built from real industrial technical reports and (ii) the end-t… |
| 59.6 | [Learning What to Fail On: Failure-Mode Contextual Bandits for Adversarial Data Curation](http://arxiv.org/abs/2608.18681v1) | We introduce a failure-aware adversarial retrieval-augmented framework for improving robustness in natural language understanding.… |
| 58.8 | [The Imitation Game: When LLMs Learn to Reason Like Programs via Code-Centric Reasoning Data Synthesis](http://arxiv.org/abs/2609.16076v1) | To bridge this gap, we propose MIMIC, a framework that leverages executable code as a rigorous medium for reasoning data synthesis.… |
| 58.4 | [Well-Specified and Well-Tested? Assessing SWE-bench-Style Datasets with a Self-Evolving Agent](http://arxiv.org/abs/2610.39417v1) | Inspired by this, we propose ExLabel, an LLM-based, self-evolving, experience-guided assistant for task clarity and test coverage assessment in SWE da… |
| 57.6 | [KMGen: A Skill-based Approach for Synthetic Individual Patient Data Generation](http://arxiv.org/abs/2608.22618v1) | We introduce KMGen, the first end-to-end framework that (i) fully automates KM curve extraction at accuracy competitive with human-guided tools, and (… |
| 57.0 | [DKL: Decoupled Knowledge Learning for Instruction-Tuned Language Models](http://arxiv.org/abs/2609.02685v1) | In this work, we propose DKL-Decoupled Knowledge Learning for Instruction-Tuned Language Models.… |
| 56.8 | [CasePlay: Self-Play Reinforcement Learning from Case Reports for Medical Reasoning](http://arxiv.org/abs/2610.24952v1) | We study whether document-grounded self-play can convert these reports into effective RL training signals.… |
| 56.4 | [SAFE-Mem: Secure Agentic Memory with Fine-Grained Authority and Governance-Aware Retrieval](http://arxiv.org/abs/2610.22959v2) | To address these limitations, we propose SAFE-Mem, a fine-grained memory governance framework that improves security while preserving legitimate memor… |
| 55.8 | [Evolve on Graph: Self-Evolving Knowledge Graph Reasoning without Question Annotations](http://arxiv.org/abs/2610.35598v1) | We propose Evolve on Graph (EoG), a framework that lets an LLM self-evolve its graph reasoning without any QA annotations.… |

### Data Ingestion Strategies

| Score | Paper | Idea |
| --- | --- | --- |
| 71.0 | [SpatialChain: A Benchmark for Auditing Spatial Reasoning Faithfulness in VLMs](http://arxiv.org/abs/2610.06413v1) | We introduce SpatialChain, a dataset of 28,350 training and 899 test examples pairing spatially-oriented GQA questions with scene-graph-grounded reaso… |
| 70.0 | [Behavior2Trip: Towards Personalized Travel Planning via User Behavior Trajectory](http://arxiv.org/abs/2608.26807v1) | To bridge this gap, we introduce a new task, Behavior-Aware Travel Planning, which infers user preferences directly from past behaviors and generates … |
| 69.8 | [GTA-RAG: Graph-Trajectory-Augmented Reinforcement Learning for Multi-Turn Retrieval-Augmented Reasoning](http://arxiv.org/abs/2608.22479v1) | We present GTA-RAG, a graph-trajectory-augmented RL framework for multi-turn retrieval-augmented reasoning.… |
| 69.8 | [Hybrid Reasoning Agents](http://arxiv.org/abs/2610.18861v1) | How can we build an agent that thinks beyond human language, yet still acts through it in the external environment?… |
| 68.6 | [Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding](http://arxiv.org/abs/2609.12243v1) | To address this, we introduce Chopthin-Consensus Power Sampling (CCPS).… |
| 68.2 | [FastBench: Can Streaming VLMs Perceive High-Dynamic Real-World Streams?](http://arxiv.org/abs/2610.12427v1) | We introduce FastBench to evaluate high-dynamic perception in real-world video streams.… |
| 67.8 | [Drive the Thoughts: Runtime Monitoring of VLA Reasoning-Trajectory Consistency](http://arxiv.org/abs/2608.29583v1) | Leveraging this potential, we propose integrating a CoT-trajectory consistency check into a runtime monitor.… |
| 67.8 | [Where Root Cause Analysis Fails: A Retrieval-Reranking Decomposition](http://arxiv.org/abs/2609.36686v1) | In this work, we introduce a retrieval-reranking decomposition and audit four well-known benchmarks to expose this blind spot.… |
| 66.6 | [SEAD: A State-Based Perspective on Attack and Defense in Tool-Using Agents](http://arxiv.org/abs/2609.34518v1) | We formulate attack and defense as partially observed state control in SEAD, deriving their design requirements from this shared execution process.… |
| 66.4 | [Beyond Factual Knowledge: Benchmarking and Learning Step-Level Procedural Rule Reasoning in Large Language Models](http://arxiv.org/abs/2608.22753v1) | To evaluate this capability, we introduce RuleWorld, a large-scale benchmark that reformulates rules as globally reusable abstract units rather than i… |
| 66.4 | [TALK-Dem: Benchmarking Embodied Task Planning under Dementia-Associated Communication Patterns](http://arxiv.org/abs/2609.38371v1) | To mitigate this issue, we proposed the Context-Aware Retrieval from Experience (CARE) method, which retrieves relevant previously resolved tasks to p… |
| 66.2 | [Seek: Self-Evaluative Exploration for Knowledge Retrieval](http://arxiv.org/abs/2609.28980v1) | We introduce Seek, Self-Evaluative Exploration for Knowledge Retrieval, a training-free framework that addresses this limitation through iterative cor… |
| 65.8 | [Test-time Calibration Learning for Large Language Model Reasoning](http://arxiv.org/abs/2610.02695v1) | To address this challenge, we propose Test-Time Calibration Learning (TTCL), a label-free framework that jointly adapts reasoning accuracy and verbali… |
| 65.8 | [RRSI: Regularized Recursive Self-Improvement of Agent Harnesses](http://arxiv.org/abs/2609.24972v2) | We introduce Regularized Recursive Self-Improvement of Agent Harnesses (RRSI), which incorporates the principles of regularizations into harness self-… |
| 65.8 | [SEER: Long-Context Reasoning via Selective Visual-Text Compression](http://arxiv.org/abs/2608.15962v1) | We present SEER, a framework that learns to select query-relevant images through visual scanning and retrieve textual content only where needed, combi… |

## Where quality disagrees with keyword relevance

High relevance but low quality (keyword-heavy, weak evidence):

| Paper | relevance | quality |
| --- | --- | --- |
| [MidTool: Mid-training Data Synthesis for Agentic Tool Use](http://arxiv.org/abs/2608.20314v1) | 34 | 39.0 |
| [Test-Time Confidence Adaptation under RL-Induced Trajectory Shift](http://arxiv.org/abs/2610.35500v1) | 26 | 36.0 |
| [Improving Synthetic Data Generation for Argument Mining via Adversarial Reinforcement Learning](http://arxiv.org/abs/2610.07699v1) | 25 | 38.0 |
| [SPADE: Self-Play in Adaptive Synthetic Executable Environments](http://arxiv.org/abs/2608.19197v2) | 25 | 45.0 |
| [ASCENT: Online Test-Time Training of Long-Horizon Agents via Self-Distillation of Verified Experience](http://arxiv.org/abs/2610.05303v1) | 20 | 28.0 |
| [Qwen-Planner-Agent: A Closed-Loop AI-for-AI Framework for Real-World Mobile Planner Agents](http://arxiv.org/abs/2609.29892v1) | 23 | 44.0 |
| [Scaling Agentic Data for Long-Horizon Terminal Intelligence](http://arxiv.org/abs/2610.13166v1) | 21 | 34.0 |
| [DataCanvas-EDU: An Agentic Framework for Instructor-Guided Synthetic Data Generation in Business Analytics Education](http://arxiv.org/abs/2609.19617v1) | 26 | 60.0 |

High quality but modest relevance (strong empirical work, weaker topical keyword match):

| Paper | relevance | quality |
| --- | --- | --- |
| [Hybrid Reasoning Agents](http://arxiv.org/abs/2610.18861v1) | 8 | 69.8 |
| [Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding](http://arxiv.org/abs/2609.12243v1) | 6 | 68.6 |
| [Where Root Cause Analysis Fails: A Retrieval-Reranking Decomposition](http://arxiv.org/abs/2609.36686v1) | 8 | 67.8 |
| [RRSI: Regularized Recursive Self-Improvement of Agent Harnesses](http://arxiv.org/abs/2609.24972v2) | 8 | 65.8 |
| [SEER: Long-Context Reasoning via Selective Visual-Text Compression](http://arxiv.org/abs/2608.15962v1) | 8 | 65.8 |
| [Route Me If You Can: A Benchmark for Query Reformulation Selection](http://arxiv.org/abs/2609.14885v1) | 7 | 65.2 |
| [Adaptive Critical Token-Aware Retrieval for Repository-Level Code Generation](http://arxiv.org/abs/2609.01601v1) | 8 | 64.8 |
| [Safe, Persistent, and Evolving Agent Harness for Understanding Partially Observable Worlds](http://arxiv.org/abs/2610.11552v1) | 7 | 64.2 |

## Limitations

- No citations, venue reputation, reviewer scores, or full-text reading; a paper with a polished abstract can score well regardless of correctness.
- Regex hits are coarse: mentioning a benchmark is not the same as releasing one, and `sota_claim` is a self-report.
- `peer_reviewed` / `revised` are sparse in this corpus (about 2% and 9%), so maturity mostly separates a small minority.
- The rule is deliberately configurable: edit `config/quality_ranking.yaml` and re-run `python scripts/rank_papers.py` to change weights or thresholds.

Full per-paper scores and signal hits are in `data/papers.json`; the complete index is [`PAPERS.md`](PAPERS.md), and the Chinese review is [`PAPER_IDEAS_ZH.md`](PAPER_IDEAS_ZH.md).
