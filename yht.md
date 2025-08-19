# 模型蒸馏
## 蒸馏阶段
### 蒸馏推理阶段
#### [Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision](https://arxiv.org/pdf/2312.09390) Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, Jeff Wu：用教师输出的弱监督信号直接对学生进行微调
#### [Alice: Proactive Learning with Teacher's Demonstrations for Weak-to-Strong Generalization Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/pdf/2504.07316) Shujin Wu1,2∗ Cheng Qian1 Yi R. (May) Fung1 Paul Pu Liang3 Heng Ji1 1University of Illinois Urbana-Champaign 2University of Southern California 3Massachusetts Institute of Technology :学生模型根据教师模型提供的不确定性表达来达到蒸馏效果
## 蒸馏架构
### 跨模型架构蒸馏
#### [Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs](https://openreview.net/pdf?id=bwRxXiGO9A)  Nicolas Boizard,Kevin El Haddad,Céline Hudelot,Pierre Colombo,Equall.ai:直接匹配不同Tokenizers的输出分布的形状
#### [Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation](https://arxiv.org/pdf/2412.14528) Xiao Cui1*, Mo Zhu2*, Yulei Qin3, Liang Xie2,4, Wengang Zhou1, Houqiang Li1:对OT进行扩展到多层级，兼顾了上下文的关系
## 蒸馏效率优化
### 稀疏蒸馏
#### [Sparse Logit Sampling: Accelerating Knowledge Distillation in LLM]（https://arxiv.org/pdf/2503.16870）  Anshumann，MohdAbbasZaidi，Akhil Kedia，Jinwoo Ahn，Taehwak Kwon，KangwookLee，HaejunLee，JoohyungLee:仅对教师模型的Top-K logits进行计算loss
## Agent蒸馏
#### [AGENTDISTILL: TRAINING-FREE AGENT DISTILLATION WITH  GENERALIZABLE MCP BOXES](https://arxiv.org/pdf/2506.14728) Jiahao Qiu, Xinzhe Juan, Yimin Wang, Ling Yang, Xuan Qi, Tongcheng Zhang, Jiacheng Guo,Yifu Lu, Zixin Yao, Hongru Wang, Shilong Liu, Xun Jiang, Liu Leqi, Mengdi Wang:通过直接将教师Agent的策略和知识打包成"MCP Box"，实现无需训练的Agent蒸馏

# 数学推理
## Evaluation Dataset
- [**MAWPS: A Math Word Problem Repository**](https://aclanthology.org/N16-1136.pdf) *Rik Koncel-Kedziorski, Subhro Roy, Aida Amini,Nate Kushman,Hannaneh Hajishirzi* NAACL 2016.
- [**Program Induction by Rationale Generation:Learning to Solve and Explain Algebraic Word Problems**](https://arxiv.org/pdf/1705.04146) *Wang Ling, Dani Yogatama, Chris Dyer, Phil Blunsom* IJCAI 2017.
- [**Training Verifiers to Solve Math Word Problems**](https://arxiv.org/pdf/2110.14168) *Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman* NeurIPS 2021.
- [**Measuring Mathematical Problem Solving With the MATH Dataset**](https://arxiv.org/abs/2103.03874) *Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt*NeurIPS Dataset and Benchmark 2021.
- [**MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics**](https://arxiv.org/abs/2109.00110) *Kunhao Zheng, Jesse Michael Han, Stanislas Polut*ICLR 2022.
- [**Lila: A Unified Benchmark for Mathematical Reasoning**](https://arxiv.org/abs/2210.17517v2) *Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, Ashwin Kalyana*EMNLP 2022.
- [**ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning**](https://arxiv.org/abs/2203.10244) *Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, Enamul Hoque*ACL(finding) 2022.（数据集中既有人类编写问题，也有根据人类编写的图标摘要生成的问题）
- [**UniGeo: Unifying Geometry Logical Reasoning via Reformulating Mathematical Expression**](https://arxiv.org/abs/2212.02746) *Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, Xiaodan Liang*EMNLP 2022.
- [**TheoremQA: A Theorem-driven Question Answering dataset**](https://arxiv.org/abs/2305.12524) *Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, Tony Xia*EMNLP 2023.
- [**CMATH: Can Your Language Model Pass Chinese Elementary School Math Test?**](https://arxiv.org/abs/2306.16636) *Tianwen Wei, Jian Luan, Wei Liu, Shuang Dong, Bin Wang*Preprint 2023.
- [**LeanDojo: Theorem Proving with Retrieval-Augmented Language Models**](https://arxiv.org/abs/2306.15626) *Kaiyu Yang, Aidan M. Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan J. Prenger, Animashree Anandkumar*NeurIPS 2023.
- [**OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems**](https://arxiv.org/abs/2402.14008) *Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, Maosong Sun* ACL 2024.
- [**OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text**](https://openreview.net/pdf?id=jKHmjlpViu) *Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, Jimmy Ba* ICLR 2024.
- [**AIME24**](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) 该数据集包含来自 2024 年美国数学邀请赛 (American Invitational Mathematics Examination, AIME) 的题目。AIME 是一项以其极具挑战性的数学问题而闻名的、享有盛誉的高中数学竞赛。
- [**MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts**](https://arxiv.org/abs/2310.02255) *Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, Jianfeng Gao*ICLR 2024.
- [**MathPile: A Billion-Token-Scale Pretraining Corpus for Math**](https://arxiv.org/abs/2312.17120) *Zengzhi Wang, Xuefeng Li, Rui Xia, Pengfei Liu* NeurIPS 2024.
- [**MathOdyssey: Benchmarking Mathematical Problem-Solving Skills in Large Language Models Using Odyssey Math Data**](https://arxiv.org/pdf/2406.18321) *Meng Fang, Xiangpeng Wan, Fei Lu, Fei Xing, Kai Zou* Preprint 2024.
- [**MuggleMath: Assessing the Impact of Query and Response Augmentation on Math Reasoning**](https://arxiv.org/abs/2310.05506v3) *Chengpeng Li, Zheng Yuan, Hongyi Yuan, Guanting Dong, Keming Lu, Jiancan Wu, Chuanqi Tan, Xiang Wang, Chang Zhou.* ACL 2024.
- [**MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs**](https://arxiv.org/abs/2402.16352) *Zimu Lu, Aojun Zhou, Houxing Ren, Ke Wang, Weikang Shi, Junting Pan, Mingjie Zhan, Hongsheng Li.* ACL 2024.
- [**MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models**](https://arxiv.org/abs/2309.12284) *Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, Weiyang Liu.* ICLR 2024.
- [**Augmenting Math Word Problems via Iterative Question Composing**](https://arxiv.org/abs/2401.09003) *Haoxiong Liu, Yifan Zhang, Yifan Luo, Andrew Chi-Chih Yao.* DPFM@ICLR 2024.
- [**DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning**](https://arxiv.org/pdf/2504.11456) *Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong Yu*. Preprint 2025
- [**MegaMath: Pushing the Limits of Open Math Corpora**](https://arxiv.org/abs/2504.02807) *Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong YuFan Zhou, Zengzhi Wang, Nikhil Ranjan, Zhoujun Cheng, Liping Tang, Guowei He, Zhengzhong Liu, Eric P. Xing*. Preprint 2025
- [**MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts**](https://arxiv.org/pdf/2502.20808) *Peijie Wang, Zhong-Zhi Li, Fei Yin, Dekang Ran, Cheng-Lin Liu* CVPR 2025.

## Synthetic Dataset
- [**Are NLP Models really able to Solve Simple Math Word Problems?**](https://arxiv.org/pdf/2103.07191) *Arkil Patel, Satwik Bhattamishra, Navin Goyal* NeurIPS 2021.
- [**Llemma: An Open Language Model For Mathematics**](https://arxiv.org/abs/2310.10631) *Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen Marcus McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, Sean Welleck* ICLR 2024.
- [**MAMMOTH: BUILDING MATH GENERALIST MODELS THROUGH HYBRID INSTRUCTION TUNING**](https://arxiv.org/abs/2309.05653) *Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, Wenhu Chen*ICLR 2024.
- [**MuggleMath: Assessing the Impact of Query and Response Augmentation on Math Reasoning**](https://arxiv.org/abs/2310.05506v3) *Chengpeng Li, Zheng Yuan, Hongyi Yuan, Guanting Dong, Keming Lu, Jiancan Wu, Chuanqi Tan, Xiang Wang, Chang Zhou.* ACL 2024.（根据种子，生成了部分数据，进行了数据增强）
- [**MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs**](https://arxiv.org/abs/2402.16352) *Zimu Lu, Aojun Zhou, Houxing Ren, Ke Wang, Weikang Shi, Junting Pan, Mingjie Zhan, Hongsheng Li.* ACL 2024.（根据种子，生成了部分数据，进行了数据增强）
- Ashvini Jindal. 2023. Arithmo-mistral-7b: Mathematical reasoning model. Hugging Face.
- [**MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models**](https://arxiv.org/abs/2309.12284) *Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, Weiyang Liu.* ICLR 2024.（根据种子，生成了部分数据，进行了数据增强）
- [**Augmenting Math Word Problems via Iterative Question Composing**](https://arxiv.org/abs/2401.09003) *Haoxiong Liu, Yifan Zhang, Yifan Luo, Andrew Chi-Chih Yao.* DPFM@ICLR 2024.（根据种子，生成了部分数据，进行了数据增强）
- OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset. NeurIPS 2024.（种子数据+生成数据）
- [**Distilling LLMs' Decomposition Abilities into Compact Language Models**](https://arxiv.org/abs/2402.01812) *Denis Tarasov, Kumar Shridhar.* AutoRL@ICML 2024.
- [**RMath: A Logic Reasoning-Focused Datasets Toward Mathematical Multistep Reasoning Tasks**](https://ojs.aaai.org/index.php/AAAI/article/view/34585) *Ziyi Hu, Jun Liu, Zhongzhi Liu, Yuzhong Liu, Zheng Xie, Yiping Song* AAAI 2025.
- OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data. Arxiv 2025.(合成数据+真实数据）
- OpenThoughts: Data Recipes for Reasoning Models. Arxiv 2025.（生成思维连）
- [**DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning**](https://arxiv.org/pdf/2504.11456) *Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong Yu*. Preprint 2025（有合成数据补充）
- [**MegaMath: Pushing the Limits of Open Math Corpora**](https://arxiv.org/abs/2504.02807) *Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong YuFan Zhou, Zengzhi Wang, Nikhil Ranjan, Zhoujun Cheng, Liping Tang, Guowei He, Zhengzhong Liu, Eric P. Xing*. Preprint 2025（有合成数据补充）


# 代码生成
## Evaluation Dataset
- [**Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task**](https://arxiv.org/pdf/1809.08887) *Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, Dragomir Radev* EMNLP 2018.
- [**CodeSearchNet Challenge: Evaluating the State of Semantic Code Search**](https://arxiv.org/abs/1909.09436) *Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, Marc Brockschmidt*Preprint 2019.
- [**Measuring Coding Challenge Competence With APPS**](https://arxiv.org/abs/2105.09938) *Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, Jacob Steinhardt* NeurIPS 2021
- [**CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion**](https://arxiv.org/abs/2310.11248) *Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, Bing Xiang*NeurIPS 2023.
- [**RepoFusion: Training Code Models to Understand Your Repository**](https://arxiv.org/abs/2306.10998) *Disha Shrivastava, Denis Kocetkov, Harm de Vries, Dzmitry Bahdanau, Torsten Scholak*Preprint 2023.
- [**DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation**](https://arxiv.org/abs/2211.11501) *Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Scott Wen-tau Yih, Daniel Fried, Sida Wang, Tao Yu* ICML 2023
- [**CodeS: Natural Language to Code Repository via Multi-Layer Sketch**](https://arxiv.org/html/2403.16443) *	Daoguang Zan, Ailun Yu, Wei Liu, Dong Chen, Bo Shen, Wei Li, Yafen Yao, Yongshun Gong, Xiaolin Chen, Bei Guan, Zhiguang Yang, Yongji Wang, Qianxiang Wang, Lizhen Cui*Preprint 2024.（ai辅助）
- [**EvoCodeBench: An Evolving Code Generation Benchmark Aligned with Real-World Code Repositories**](https://arxiv.org/abs/2404.00599) *Jia Li, Ge Li, Xuanming Zhang, Yihong Dong, Zhi Jin*Preprint 2024.
- [**RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems**](https://arxiv.org/abs/2306.03091) *Tianyang Liu, Canwen Xu, Julian McAuley* ICLR 2024
## Synthetic Dataset
- [**Code Alpaca: An Instruction-following LLaMA Model trained on code generation instructions**](https://github.com/sahil280114/codealpaca) *Sahil Chaudhary*. GitHub 2023.
- [**WizardCoder: Empowering Code Large Language Models with Evol-Instruct**](https://arxiv.org/abs/2306.08568) *Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, Daxin Jiang.* ICLR 2024.
- [**WaveCoder: Widespread And Versatile Enhancement For Code Large Language Models By Instruction Tuning**](https://arxiv.org/abs/2312.14187) *Zhaojian Yu, Xin Zhang, Ning Shang, Yangyu Huang, Can Xu, Yishujie Zhao, Wenxiang Hu, Qiufeng Yin.* ACL 2024.
- [**Magicoder: Empowering Code Generation with OSS-Instruct**](https://arxiv.org/abs/2312.02120) *Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, Lingming Zhang.* ICML 2024.
## Methods
- [**Mapping Language to Code in Programmatic Context**](https://arxiv.org/pdf/1808.09588) *Srinivasan Iyer, Ioannis Konstas, Alvin Cheung, Luke Zettlemoyer* EMNLP 2018.
- [**CodeBERT: A Pre-Trained Model for Programming and Natural Languages**](https://arxiv.org/pdf/2002.08155) *Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, Ming Zhou* EMNLP 2020.
- [**Evaluating Large Language Models Trained on Code**](https://arxiv.org/abs/2107.03374) *Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, Wojciech Zaremba* Preprint 2021
- [**Program Synthesis with Large Language Models**](https://arxiv.org/abs/2108.07732) *Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, Charles Sutton* Preprint 2021
- [**CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning**](https://arxiv.org/abs/2207.01780) *Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, Steven C.H. Hoi.*  NeurIPS 2022.
- [**CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis**](https://arxiv.org/abs/2203.13474) *Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, Caiming Xiong* ICLR 2023
- [**Language Models Can Teach Themselves to Program Better**](https://arxiv.org/abs/2207.14502) *Patrick Haluptzok, Matthew Bowers, Adam Tauman Kalai.* ICLR 2023.
- [**InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback**](https://arxiv.org/abs/2306.14898) *John Yang, Akshara Prabhakar, Karthik Narasimhan, Shunyu Yao.* Arxiv 2023.
- [**CoderEval: A Benchmark of Pragmatic Code Generation with Generative Pre-trained Models**](https://arxiv.org/abs/2302.00288) *Hao Yu, Bo Shen, Dezhi Ran, Jiaxin Zhang, Qi Zhang, Yuchi Ma, Guangtai Liang, Ying Li, Qianxiang Wang, Tao Xie* ICSE 2024
- [**CodecLM: Aligning Language Models with Tailored Synthetic Data**](https://arxiv.org/abs/2404.05875) *Zifeng Wang, Chun-Liang Li, Vincent Perot, Long T. Le, Jin Miao, Zizhao Zhang, Chen-Yu Lee, Tomas Pfister.* Findings of NAACL 2024.
- [**Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models**](https://arxiv.org/abs/2407.21077) *Somshubra Majumdar, Vahid Noroozi, Sean Narenthiran, Aleksander Ficek, Jagadeesh Balam, Boris Ginsburg.* Arxiv 2024.
- [**Learning Performance-Improving Code Edits**](https://arxiv.org/abs/2302.07867) *Alexander Shypula, Aman Madaan, Yimeng Zeng, Uri Alon, Jacob Gardner, Milad Hashemi, Graham Neubig, Parthasarathy Ranganathan, Osbert Bastani, Amir Yazdanbakhsh.* ICLR 2024.
- [**InverseCoder: Unleashing the Power of Instruction-Tuned Code LLMs with Inverse-Instruct**](https://arxiv.org/abs/2407.05700) *Yutong Wu, Di Huang, Wenxuan Shi, Wei Wang, Lingzhe Gao, Shihao Liu, Ziyuan Nan, Kaizhao Yuan, Rui Zhang, Xishan Zhang, Zidong Du, Qi Guo, Yewen Pu, Dawei Yin, Xing Hu, Yunji Chen.* Arxiv 2024.
- [**OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement**](https://arxiv.org/abs/2402.14658) *Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, Xiang Yue.* Arxiv 2024.
- [**AutoCoder: Enhancing Code Large Language Model with AIEV-Instruct**](https://arxiv.org/abs/2405.14906) *Bin Lei, Yuchen Li, Qiuwu Chen.* Arxiv 2024.
- [**How Do Your Code LLMs Perform? Empowering Code Instruction Tuning with High-Quality Data**](https://www.arxiv.org/abs/2409.03810) *Yejie Wang, Keqing He, Dayuan Fu, Zhuoma Gongque, Heyang Xu, Yanxu Chen, Zhexu Wang, Yujia Fu, Guanting Dong, Muxi Diao, Jingang Wang, Mengdi Zhang, Xunliang Cai, Weiran Xu.* Arxiv 2024.
- [**SelfCodeAlign: Self-Alignment for Code Generation**](https://arxiv.org/abs/2410.24198) *Yuxiang Wei, Federico Cassano, Jiawei Liu, Yifeng Ding, Naman Jain, Zachary Mueller, Harm de Vries, Leandro von Werra, Arjun Guha, Lingming Zhang.* Arxiv 2024.
- OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement. Findings of ACL 2024.
- CodeEvo: Interaction-Driven Synthesis of Code-centric Data through Hybrid and Iterative Feedback. Arxiv 2025.

