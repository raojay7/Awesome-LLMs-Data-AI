# Awesome-LLMs-Data-AI
# A Survey on Data-centric AI

<div align="center">

[![LICENSE](https://img.shields.io/github/license/wasiahmad/Awesome-LLM-Synthetic-Data-Generation)](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data-Generation/blob/main/LICENSE)
![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
<!-- ![license](https://img.shields.io/bower/l/bootstrap?style=plastic) -->

</div>

This repo includes papers and blogs about the survey on Data-centric AI of LLMs.

Thanks for all the great contributors on GitHub!🔥⚡🔥

- To Do
- [x] 每个人完成各自标记的文献总结，要求，通过找到自己的名字，如（WZY）来完成各自的部分：
- [ ] 1. 一句话概况各个paper的idea，形成 paper名及idea概况的 Map（第一个映射）
- [ ] 2. 各自子标题下归类总结一类paper做了什么，形成类别及paper名的 Map（第二个映射）
- [ ] 3. 通过已有文献关键词，数据合成教程和之前给的seed paper（都是主题相关的）尽可能调研近3年已有的paper （比较出名的，比如顶会或者引用高）
- [ ] 4. 确定这篇论文的出版物，如： [**Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision**](https://arxiv.org/abs/2305.03047) *Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, Chuang Gan.* NeurIPS 2023.
- [ ] 5. 把这部分内容在各自的md文件更新

survey 总体结构
1. Surveys-> related work
2. Methods-> How to get data?
3. Stages-> How to use?
4. Application Areas-> Where to use?
## Contents

- [A Survey on Data-centric AI](#a-survey-on-data-centric-ai)
  - [Contents](#contents)
  - [1. Surveys](#1-surveys)
  - [2. Methods](#2-methods)
    - [2.1. for Generation](#21-for-generation)
    - [2.2. for Filter](#22-for-filter)
  - [3. Stages](#3-stages)
    - [3.1. pretrain](#31-pretrain)
    - [3.2. sft](#32-sft)
    - [3.3. RL](#33-rl)     
  - [4. Application Areas](#4-application-areas)
    - [4.1. Mathematical Reasoning](#41-mathematical-reasoning)
    - [4.2. Code Generation](#42-code-generation)
    - [4.3. Safety and Alignment](#43-safety-and-alignment)
    - [4.4. Long Context](#44-long-context)
    - [4.5. Agent and Tool Use](#45-agent-and-tool-use)
    - [4.6. Vision and Language](#46-vision-and-language)




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

### Blogs

- [**Synthetic dataset generation techniques: Self-Instruct**](https://huggingface.co/blog/davanstrien/self-instruct) *Daniel van Strien.* 2024
- [**LLM-Driven Synthetic Data Generation, Curation & Evaluation**](https://cobusgreyling.medium.com/llm-driven-synthetic-data-generation-curation-evaluation-33731e33b525) *Cobus Greyling.* 2024
- [**The Rise of Agentic Data Generation**](https://huggingface.co/blog/mlabonne/agentic-datagen) *Maxime Labonne.* 2024
- https://blog.csdn.net/qq_43688587/article/details/148533722 
## 2. Methods
How to get data?
### 2.1. for Generation 
Data-generation method overview
#### 2.1.1 highlight
#### 2.1.2 Sampling-based
##### KD (YHT)
##### sequence-level KD (YHT)
- [**Impossible Distillation for Paraphrasing and Summarization: How to Make High-quality Lemonade out of Small, Low-quality Models**](https://arxiv.org/abs/2305.16635) *Jaehun Jung, Peter West, Liwei Jiang, Faeze Brahman, Ximing Lu, Jillian Fisher, Taylor Sorensen, Yejin Choi.* NAACL 2024.

##### task data from LMs (DYHP)
- [**TarGEN: Targeted Data Generation with Large Language Models**](https://arxiv.org/abs/2310.17876) *Himanshu Gupta, Kevin Scaria, Ujjwala Anantheswaran, Shreyas Verma, Mihir Parmar, Saurabh Arjun Sawant, Chitta Baral, Swaroop Mishra.* COLM 2024.
- [**Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation**](https://arxiv.org/abs/2402.18334) *Nihal V. Nayak, Yiyang Nan, Avi Trost, Stephen H. Bach* ACL Findings 2024.

##### from scratch(new task, e.g., self-instruct)从0开始合成数据 (DYHP)
- Absolute Zero: Reinforced Self-play Reasoning with Zero Data
- Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement
- Synthetic Data RL: Task Definition Is All You Need
- [**Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models**](https://arxiv.org/abs/2402.13064) *Haoran Li, Qingxiu Dong, Zhengyang Tang, Chaojun Wang, Xingxing Zhang, Haoyang Huang, Shaohan Huang, Xiaolong Huang, Zeqiang Huang, Dongdong Zhang, Yuxian Gu, Xin Cheng, Xun Wang, Si-Qing Chen, Li Dong, Wei Lu, Zhifang Sui, Benyou Wang, Wai Lam, Furu Wei.* Arxiv 2024.
- [**Self-instruct: Aligning language models with self-generated instructions**](https://arxiv.org/abs/2212.10560) *Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi.* ACL 2023.
- [**WizardLM: Empowering Large Language Models to Follow Complex Instructions**](https://arxiv.org/abs/2304.12244) *Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Daxin Jiang.* ICLR 2024.
- [**Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing**](https://arxiv.org/abs/2406.08464) *Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, Bill Yuchen Lin* NeurIPS 2024.
####  2.1.3 generation Back-translation (SJJ)
  - MT
  - application to MATH
####  2.1.4 Transformation of existing data (task Related datasets or documents) (WZY)
##### Knowledge Graph
##### Extract instruction data from the web
- MAmmoTH2: Scaling Instructions from the Web (Instruction generation based on L0 data)
- [**On the Diversity of Synthetic Data and its Impact on Training Large Language Models**](https://arxiv.org/abs/2410.15226)  *Hao Chen, Abdul Waheed, Xiang Li, Yidong Wang, Jindong Wang, Bhiksha Raj, Marah I. Abdin* Arxiv 2024.
##### Rephrasing documents for pretraining
- [**Scaling Synthetic Data Creation with 1,000,000,000 Personas**](https://arxiv.org/abs/2406.20094) *Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, Dong Yu.* NeurIPS 2024.
- [**Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling**](https://arxiv.org/abs/2401.16380) *Pratyush Maini, Skyler Seto, He Bai, David Grangier, Yizhe Zhang, Navdeep Jaitly.* ACL 2024.
- [**Source2Synth: Synthetic Data Generation and Curation Grounded in Real Data Sources**](https://arxiv.org/abs/2409.08239) *Alisia Lupidi, Carlos Gemmell, Nicola Cancedda, Jane Dwivedi-Yu, Jason Weston, Jakob Foerster, Roberta Raileanu, Maria Lomeli* Arxiv 2024.
#### 2.1.5 Human-Al collaboration 
##### self-generation (MHS)
  - [**STaR: Bootstrapping Reasoning With Reasoning**](https://arxiv.org/abs/2203.14465) *Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman.* NeurIPS 2022.
  - [**Symbolic Knowledge Distillation: from General Language Models to Commonsense Models**](https://arxiv.org/abs/2110.07178) *Peter West, Chandra Bhagavatula, Jack Hessel, Jena D. Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, Yejin Choi.* NAACL 2022.
  - [**Generating Training Data with Language Models: Towards Zero-Shot Language Understanding**](https://arxiv.org/abs/2202.04538) *Yu Meng, Jiaxin Huang, Yu Zhang, Jiawei Han.* NeurIPS 2022.
  - [**ZeroGen: Efficient Zero-shot Learning via Dataset Generation**](https://arxiv.org/abs/2202.07922) *Jiacheng Ye, Jiahui Gao, Qintong Li, Hang Xu, Jiangtao Feng, Zhiyong Wu, Tao Yu, Lingpeng Kong.* EMNLP 2022.
  - [**Large Language Models Can Self-Improve**](https://aclanthology.org/2023.emnlp-main.67/) *Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, Jiawei Han.* EMNLP 2023.
  - [**CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society**](https://arxiv.org/abs/2303.17760) *Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, Bernard Ghanem.* NeurIPS 2023.
  - [**Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models**](https://arxiv.org/abs/2401.01335) *Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu.* ICML 2024.
  - [**Self-Rewarding Language Models.**](https://arxiv.org/abs/2401.10020) *Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston.* Arxiv 2024.
  - [**Automatic Instruction Evolving for Large Language Models**](https://arxiv.org/abs/2406.00770) *Weihao Zeng, Can Xu, Yingxiu Zhao, Jian-Guang Lou, Weizhu Chen.* Arxiv 2024.
  - [**Self-playing Adversarial Language Game Enhances LLM Reasoning**](https://arxiv.org/abs/2404.10642) *Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du* Arxiv 2024.

##### Focus on Error (WZY)
- APT: Improving Specialist LLM Performance with Weakness Case Acquisition and Iterative Preference Training. Findings of ACL 2025.
##### reward modeling (YZH)
  
 
#### 2.1.6 Symbolic generation 
  - Symbolic (MATH) (SJJ)

#### 2.1.7 Scaling Laws 
- Training compute-optimal large language models
- Scaling Language Models Methods, Analysis & Insights from Training Gopher
- Scaling Laws for Neural Language Models
- Performance Law of Large Language Models
- Entropy law: The story behind data compression and llm performance
- Revisiting Scaling Laws for Language Models: The Role of Data Quality and Training Strategies. ACL 2025.


### 2.2. for Filter 
Approaches to data filtering
Measure metric 
(WSH ALL)
<img width="1079" height="476" alt="image" src="https://github.com/user-attachments/assets/a21a2597-f4b7-49f3-9255-ab0204d63e6f" />
<img width="1794" height="861" alt="image" src="https://github.com/user-attachments/assets/2ef0571b-b1cf-48eb-8fbe-60cd52007a66" />
#### 2.2.1 Diversity filtering
  - surface-level heuristics 
    - Rouge-L (Self-Instruct, Impossible Distillation))
    - Embedding similarity (QDIT, DiverseEvol, DEITA)
    - Semantic tags (instag)
  -  loss gradients [https://nvlabs.github.io/prismatic-synthesis/]
#### 2.2.2 Quality filtering
  -  reward models
  -  Correctness (final answer verification) 

#### 2.2.3 Multi filtering
- Multi-agent collaborative data selection for efficient llm pretraining
- LIMO: less is more for reasoning
- DataMan: data manager for pre-training large language model
- Qurating: Selecting high-quality data for training language model
- DELE: data efficient LLM evaluation
- DSDM: model-aware dataset selection with datamodels
- Rethinking Data Selection at Scale: Random Selection is Almost All You Need (Analysis of the million-level instruction selection algorithm)



## 3. Stages
How to use these data (stage from pretrain to rl)?
**for Training**

### 3.1 Pretrain 
(SJJ调研预训练相关方法和现有预训练数据集)


#### 3.1.1 highlight
#### 3.1.2 Training Methods
data process
- D4: Improving LLM Pretraining via Document De-Duplication and Diversification

mix
- DoReMi Optimizing Data Mixtures Speeds Up Language Model Pretraining
- LESS: Selecting Influential Data for Targeted Instruction Tuning
- How far can camels go? exploring the state of instruction tuning on open resources
  
关注质量和多样性
- Doremi: Optimizing data mixtures speeds up language model pretraining
- SampleMix: A Sample-wise Pre-training Data Mixing Strategey by Coordinating Data Quality and Diversity
- QuaDMix: Quality-Diversity Balanced Data Selection for Efficient LLM Pretraining
- Predictive Data Selection: The Data That Predicts Is the Data That Teaches
- Recycling the Web: A Method to Enhance Pre-training Data Quality and Quantity for Language Models
- StatsMerging: Statistics-Guided Model Merging via Task-Specific Teacher Distillation
- Quality, Diversity, and Complexity in Synthetic Data
- Rethinking Data Mixture for Large Language Models: A Comprehensive Survey and New Perspectives
selection
- HKUST Predictive Data Selection: The Data That Predicts Is the Data That Teaches
- 字节 QuaDMix: Quality-Diversity Balanced Data Selection for Efficient LLM Pretraining
- 美团 SampleMix: A Sample-wise Pre-training Data Mixing Strategey by Coordinating Data Quality and Diversity
- Recycling the Web: A Method to Enhance Pre-training Data Quality and Quantity for Language Models
#### 3.1.3 Construction
(SJJ)
1. existing datasets
- GneissWeb： Recipe for producing a state-of-the-art LLM pre-training dataset having 10+ Trillion tokens, derived from 
FineWeb V1.1.0： https://huggingface.co/datasets/ibm-granite/GneissWeb
- DCLM-baseline: DCLM-baseline is a 4T token / 3B document pretraining dataset that achieves strong performance on language model benchmarks. https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
- Dolma Dataset: an open dataset of 3 trillion tokens from a diverse mix of web content, academic publications, code, books, and encyclopedic materials. https://huggingface.co/datasets/allenai/dolma
- Zyda-2: a 5 Trillion Token High-Quality Dataset with NVIDIA NeMo Curator combining a variety of data sources obtained through different processing pipelines leads to more diverse data (including dclm, fineweb-edu2, dolma-CC and Zyda-1) https://www.zyphra.com/post/building-zyda-2
- Fineweb-edu: consists of 1.3T tokens and 5.4T tokens (FineWeb-Edu-score-2) of educational web pages filtered from  FineWeb dataset. https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Chinese Fineweb Edu Dataset V2.1 is an enhanced version of the V2 dataset, designed specifically for natural language processing (NLP) tasks in the education sector.  https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1 
- Multimodal c4: An open, billion-scale corpus of images interleaved with text
- CCpdf: Building a High Quality Corpus for Visually Rich Documents from Web Crawl Data
- The RefinedWeb dataset for Falcon LLM: outperforming curated corpora with web data, and web data only
- Extracting representative subset from extensive text data for training pre-trained language models
- A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity
- Can Data Diversity Enhance Learning Generalization?
  
2. Rephrase existing text


3. Verbalize knowledge bases using LMs
 - Generate text without using LMs (e.g. formal languages) 
    - formatting
    - Symbolic generation
  - Continued Pretraining 
    - domain adaptation

### 3.2 SFT
#### 3.2.1 highlight
    - Control the style of the model’s output
    - Specialize behavior for a particular use-case
    - Feed new information to the model
#### 3.2.2 Training Methods
- REFT: Reasoning with REinforced Fine-Tuning (RFT)
- CommonIT: Commonality-Aware Instruction Tuning for Large Language Models via Data Partitions 
#### 3.2.3 Construction
  - Distillation (YHT)
  - Self-Guide (WZY)
### 3.3 RL
#### 3.2.1 highlight
    - Learn from minimal supervision
    - Learn from negative examples (e.g. harmful behavior)
    - Adapt models to their own token distribution rather than text written by others (“exposure bias”) 
#### 3.2.2 Training Methods
KTO DPO PPO等
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)
- SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training (SFT & RL Action Analysis)
- All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning
   - Synthetic Feedback (Algorithms adapt to data) (YZH)
    - LLM-as-a-judge (Prometheus)
    - preference learning
    - Flexible Criteria
    - evaluation
      - Agreement with human preferences
      - Agreement with generic benchmarks
      - Effectiveness in RL pipelines (task performance)
#### 3.2.3 Construction
- [**Constitutional AI: Harmlessness from AI Feedback**](https://arxiv.org/abs/2212.08073) *Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan.* Arxiv 2022.
- [**Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision**](https://arxiv.org/abs/2305.03047) *Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, Chuang Gan.* NeurIPS 2023.
- [**SALMON: Self-Alignment with Instructable Reward Models**](https://arxiv.org/abs/2310.05910) *Zhiqing Sun, Yikang Shen, Hongxin Zhang, Qinhong Zhou, Zhenfang Chen, David Cox, Yiming Yang, Chuang Gan.* ICLR 2024.
- [**Refined Direct Preference Optimization with Synthetic Data for Behavioral Alignment of LLMs**](https://arxiv.org/abs/2402.08005) *V´ıctor Gallego.* Arxiv 2024.
- [**Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models**](https://arxiv.org/abs/2406.13542) *Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, Jingren Zhou* ICLR 2025.
- [**Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts**](https://arxiv.org/abs/2402.16822) *Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, Roberta Raileanu.* NeurIPS 2024.


- [**West-of-N: Synthetic Preference Generation for Improved Reward Modeling**](https://arxiv.org/abs/2401.12086) *Alizée Pace, Jonathan Mallinson, Eric Malmi, Sebastian Krause, Aliaksei Severyn.* Arxiv 2024.
## 4. Application Areas
Where to use?
### 4.1. Mathematical Reasoning 
(YHT)
- [**MuggleMath: Assessing the Impact of Query and Response Augmentation on Math Reasoning**](https://arxiv.org/abs/2310.05506v3) *Chengpeng Li, Zheng Yuan, Hongyi Yuan, Guanting Dong, Keming Lu, Jiancan Wu, Chuanqi Tan, Xiang Wang, Chang Zhou.* ACL 2024.
- [**MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs**](https://arxiv.org/abs/2402.16352) *Zimu Lu, Aojun Zhou, Houxing Ren, Ke Wang, Weikang Shi, Junting Pan, Mingjie Zhan, Hongsheng Li.* ACL 2024.
- [**MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models**](https://arxiv.org/abs/2309.12284) *Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, Weiyang Liu.* ICLR 2024.
- [**Augmenting Math Word Problems via Iterative Question Composing**](https://arxiv.org/abs/2401.09003) *Haoxiong Liu, Yifan Zhang, Yifan Luo, Andrew Chi-Chih Yao.* DPFM@ICLR 2024.
- [**Distilling LLMs' Decomposition Abilities into Compact Language Models**](https://arxiv.org/abs/2402.01812) *Denis Tarasov, Kumar Shridhar.* AutoRL@ICML 2024.

### 4.2. Code Generation
(YHT)
- [**CodecLM: Aligning Language Models with Tailored Synthetic Data**](https://arxiv.org/abs/2404.05875) *Zifeng Wang, Chun-Liang Li, Vincent Perot, Long T. Le, Jin Miao, Zizhao Zhang, Chen-Yu Lee, Tomas Pfister.* Findings of NAACL 2024.
- [**CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning**](https://arxiv.org/abs/2207.01780) *Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, Steven C.H. Hoi.*  NeurIPS 2022.
- [**Language Models Can Teach Themselves to Program Better**](https://arxiv.org/abs/2207.14502) *Patrick Haluptzok, Matthew Bowers, Adam Tauman Kalai.* ICLR 2023.
- [**InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback**](https://arxiv.org/abs/2306.14898) *John Yang, Akshara Prabhakar, Karthik Narasimhan, Shunyu Yao.* Arxiv 2023.
- [**Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models**](https://arxiv.org/abs/2407.21077) *Somshubra Majumdar, Vahid Noroozi, Sean Narenthiran, Aleksander Ficek, Jagadeesh Balam, Boris Ginsburg.* Arxiv 2024.
- [**Learning Performance-Improving Code Edits**](https://arxiv.org/abs/2302.07867) *Alexander Shypula, Aman Madaan, Yimeng Zeng, Uri Alon, Jacob Gardner, Milad Hashemi, Graham Neubig, Parthasarathy Ranganathan, Osbert Bastani, Amir Yazdanbakhsh.* ICLR 2024.
 - [**WizardCoder: Empowering Code Large Language Models with Evol-Instruct**](https://arxiv.org/abs/2306.08568) *Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, Daxin Jiang.* ICLR 2024.
- [**WaveCoder: Widespread And Versatile Enhancement For Code Large Language Models By Instruction Tuning**](https://arxiv.org/abs/2312.14187) *Zhaojian Yu, Xin Zhang, Ning Shang, Yangyu Huang, Can Xu, Yishujie Zhao, Wenxiang Hu, Qiufeng Yin.* ACL 2024.
- [**Magicoder: Empowering Code Generation with OSS-Instruct**](https://arxiv.org/abs/2312.02120) *Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, Lingming Zhang.* ICML 2024.
- [**InverseCoder: Unleashing the Power of Instruction-Tuned Code LLMs with Inverse-Instruct**](https://arxiv.org/abs/2407.05700) *Yutong Wu, Di Huang, Wenxuan Shi, Wei Wang, Lingzhe Gao, Shihao Liu, Ziyuan Nan, Kaizhao Yuan, Rui Zhang, Xishan Zhang, Zidong Du, Qi Guo, Yewen Pu, Dawei Yin, Xing Hu, Yunji Chen.* Arxiv 2024.
- [**OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement**](https://arxiv.org/abs/2402.14658) *Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, Xiang Yue.* Arxiv 2024.
- [**AutoCoder: Enhancing Code Large Language Model with AIEV-Instruct**](https://arxiv.org/abs/2405.14906) *Bin Lei, Yuchen Li, Qiuwu Chen.* Arxiv 2024.
- [**How Do Your Code LLMs Perform? Empowering Code Instruction Tuning with High-Quality Data**](https://www.arxiv.org/abs/2409.03810) *Yejie Wang, Keqing He, Dayuan Fu, Zhuoma Gongque, Heyang Xu, Yanxu Chen, Zhexu Wang, Yujia Fu, Guanting Dong, Muxi Diao, Jingang Wang, Mengdi Zhang, Xunliang Cai, Weiran Xu.* Arxiv 2024.
- [**SelfCodeAlign: Self-Alignment for Code Generation**](https://arxiv.org/abs/2410.24198) *Yuxiang Wei, Federico Cassano, Jiawei Liu, Yifeng Ding, Naman Jain, Zachary Mueller, Harm de Vries, Leandro von Werra, Arjun Guha, Lingming Zhang.* Arxiv 2024.
#### 4.2.1 Datasets
- [**Synthetic-Text-To-SQL: A synthetic dataset for training language models to generate SQL queries from natural language prompts**](https://huggingface.co/datasets/gretelai/synthetic-text-to-sql) *Meyer, Yev and Emadi, Marjan and Nathawani, Dhruv and Ramaswamy, Lipika and Boyd, Kendrick and Van Segbroeck, Maarten and Grossman, Matthew and Mlocek, Piotr and Newberry, Drew.* Huggingface 2024.
- [**Open Artificial Knowledge**](https://huggingface.co/datasets/tabularisai/oak) *Vadim Borisov, Richard Schreiber.* ICML Workshop 2024.
- [**Code Alpaca: An Instruction-following LLaMA Model trained on code generation instructions**](https://github.com/sahil280114/codealpaca) *Sahil Chaudhary*. GitHub 2023.
- [**SynthPAI: A Synthetic Dataset for Personal Attribute Inference**](https://arxiv.org/abs/2406.07217) *Hanna Yukhymenko, Robin Staab, Mark Vero, Martin Vechev.* NeurIPS D&B 2024.

### 4.3. Safety and Alignment
(SJJ)
- [**Fine-tuning Language Models for Factuality**](https://arxiv.org/abs/2311.08401) *Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher D. Manning, Chelsea Finn.* Arxiv 2023.
- [**MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents**](https://arxiv.org/abs/2404.10774) *Liyan Tang, Philippe Laban, Greg Durrett.* Arxiv 2024.




### 4.4. Long Context
- [**Make Your LLM Fully Utilize the Context.**](https://arxiv.org/abs/2404.16811) *Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou.* Arxiv 2024.
- [**From Artificial Needles to Real Haystacks: Improving Retrieval Capabilities in LLMs by Finetuning on Synthetic Data**](https://arxiv.org/abs/2406.19292) *Zheyang Xiong, Vasilis Papageorgiou, Kangwook Lee, Dimitris Papailiopoulos*. Arxiv 2024.
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (system 2)
- KIMI


### 4.5. Agent and Tool Use
(MHS)
- [**Toolformer: Language Models Can Teach Themselves to Use Tools**](https://arxiv.org/abs/2302.04761) *Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom.* NeurIPS 2023.
- [**GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction**](https://arxiv.org/abs/2305.18752) *Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, Ying Shan.* Arxiv 2024.
- [**Gorilla: Large Language Model Connected with Massive APIs**](https://arxiv.org/abs/2305.15334) *Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez.* Arxiv 2023.
- [**ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases**](https://arxiv.org/abs/2306.05301) *Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, Le Sun.* Arxiv 2023.
- [**Voyager: An Open-Ended Embodied Agent with Large Language Models**](https://arxiv.org/abs/2305.16291) *Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar.* Arxiv 2023.
- [**DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows**](https://arxiv.org/abs/2402.10379) *Ajay Patel, Colin Raffel, Chris Callison-Burch.* ACL 2024.
- [**AgentInstruct: Toward Generative Teaching with Agentic Flows**](https://arxiv.org/abs/2407.03502) *Arindam Mitra, Luciano Del Corro, Guoqing Zheng, Shweti Mahajan, Dany Rouhana, Andres Codas, Yadong Lu, Wei-ge Chen, Olga Vrousgos, Corby Rosset, Fillipe Silva, Hamed Khanpour, Yash Lara, Ahmed Awadallah.* Arxiv 2024.
- [**Distilabel: An AI Feedback (AIF) Framework for Building Datasets with and for LLMs**](https://github.com/argilla-io/distilabel) *Álvaro Bartolomé Del Canto, Gabriel Martín Blázquez, Agustín Piqueres Lajarín and Daniel Vila Suero.* GitHub 2024.
- [**Fuxion: Synthetic Data Generation and Normalization Functions using Langchain + LLMs**](https://github.com/tobiadefami/fuxion)


### 4.6. Vision and Language

- [**Visual Instruction Tuning**](https://arxiv.org/abs/2304.08485) *Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee.* NeurIPS 2023.
- [**MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models**](https://arxiv.org/abs/2304.10592) *Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny.* ICLR 2024.
- [**Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond**](https://arxiv.org/abs/2308.12966) *Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, Jingren Zhou.* Arxiv 2023.
- [**G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model**](https://arxiv.org/abs/2312.11370) *Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, Lingpeng Kong.* Arxiv 2023.
- [**Enhancing Large Vision Language Models with Self-Training on Image Comprehension**](https://arxiv.org/abs/2405.19716) *Yihe Deng, Pan Lu, Fan Yin, Ziniu Hu, Sheng Shen, James Zou, Kai-Wei Chang, Wei Wang.* Arxiv 2024.
- [**LLaVA-OneVision: Easy Visual Task Transfer**](https://arxiv.org/abs/2408.03326) *Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, Chunyuan Li* Arxiv 2024.


