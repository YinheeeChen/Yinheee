---
title: Pocket-peptide课题相关的论文

published: 2025-03-30

description: ''

image: ''

tags: [Pocket-peptide, Diffusion, Flow matching]

category: '科研'

draft: false

lang: ''

---

# 论文

1. ~~**PepGLAD**~~ ~~Full-Atom Peptide Design with Geometric Latent Diffusion~~ ~~https://arxiv.org/abs/2402.13555~~

2. **SurfDock** is a surface-informed diffusion generative model for reliable and accurate protein–ligand complex prediction https://www.nature.com/articles/s41592-024-02516-y

3. **RFDiffusion** Broadly applicable and accurate protein design by integrating structure prediction networks and diffusion generative models https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1

4. Accurate structure prediction of biomolecular interactions with **AlphaFold 3** https://www.nature.com/articles/s41586-024-07487-w

5. **DiffAb** Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models for Protein Structures https://www.biorxiv.org/content/10.1101/2022.07.10.499510v5

6. FoldToken4: Consistent & Hierarchical Fold Language https://www.biorxiv.org/content/10.1101/2024.08.04.606514v2

7. **PepFlow** Full-Atom Peptide Design based on Multi-modal Flow Matching http://arxiv.org/abs/2406.00735

8. ~~SE(3) diffusion model with application to protein backbone generation~~ ~~http://arxiv.org/abs/2302.02277~~

9. ~~Efficient generation of protein pockets with PocketGen~~ ~~https://www.nature.com/articles/s42256-024-00920-9~~

10. Generalized Protein Pocket Generation with Prior-Informed Flow Matching https://arxiv.org/pdf/2409.19520

11. Anchor extension: a structure-guided approach to design cyclic peptides targeting enzyme active sites https://www.nature.com/articles/s41467-021-23609-8

|                                                                                   |               |                                                                                                                                                                                                                                                                              |
| --------------------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Paper                                                                             |               |                                                                                                                                                                                                                                                                              |
| **SurfDock,** **AlphaFold 3, DiffAb，PepFlow ，RFDiffusion**<br><br><br>            | 梁             | https://www.nature.com/articles/s41592-024-02516-y，https://www.nature.com/articles/s41586-024-07487-w，https://www.biorxiv.org/content/10.1101/2022.07.10.499510v5，<br><br>http://arxiv.org/abs/2406.00735<br><br>https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1 |
| **PepFlow, Generalized Protein Pocket Generation, Anchor extension，protein mpnn** | 陈<br><br><br> | http://arxiv.org/abs/2406.00735<br><br>https://arxiv.org/pdf/2409.19520<br><br>https://www.nature.com/articles/s41467-021-23609-8<br><br>https://www.science.org/doi/10.1126/science.add2187                                                                                 |
| **ProteinGenerator, 白军, PepFlow ，HYDRA**                                          | 孟             | https://www.nature.com/articles/s41467-021-23609-8<br><br><br>                                                                                                                                                                                                               |

# 数据集

|                                 |                                |                                                                |                                           |                                                 |
| ------------------------------- | ------------------------------ | -------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------- |
| **数据集名称**                       | **来源/用途**                      | **规模**                                                         | **相关论文**                                  | **链接**                                          |
| **PepBench**                    | **蛋白质-肽复合物**（非冗余）<br><br><br>  | - 训练集: 4,157条目，952簇<br>- 验证集: 114条目，50簇<br>- 测试集: 93条目，93簇     | PepGLAD                                   | https://zenodo.org/records/13373108<br><br><br> |
| **PepBDB**                      | **蛋白质-肽相互作用结构数据库**<br><br><br> | - 训练集: 8,434条目，1,617簇<br>- 验证集: 370条目，95簇<br>- 测试集: 190条目，190簇 | PepGLAD, Full-Atom Peptide Design         | http://huanglab.phys.hust.edu.cn/pepbdb/        |
| **ProtFrag**                    | PDB中的单体蛋白质片段（无监督数据）            | 70,645个片段                                                      | PepGLAD                                   | https://zenodo.org/records/13373108             |
| **PDBbind2020**                 | **蛋白质-配体复合物（训练和测试）**           | 363个样本（时间分割测试集）                                                | SurfDock<br><br><br>                      | http://www.pdbbind.org.cn/                      |
| **Astex Diverse Set**           | 药物样小分子性能评估                     | 85个案例                                                          | SurfDock                                  | https://www.ccdc.cam.ac.uk/                     |
| **PoseBusters Benchmark**       | 评估生成姿势的合理性和泛化性                 | V1: 428个结构；V2: 308个结构                                          | SurfDock, AlphaFold 3                     | https://github.com/maabuu/posebusters           |
| **DockGen Dataset**<br><br><br> | 测试新结合域/口袋的表现                   | - DockGen-full: 189个复合物<br>- DockGen-cluster: 85个复合物           | SurfDock                                  | -                                               |
| **DEKOIS 2.0**                  | 虚拟筛选性能评估                       | 81个蛋白质靶标                                                       | SurfDock                                  | http://www.dekois.com/                          |
| **ALDH1B1实验数据**                 | 内部药物样小分子验证                     | 37,410个小分子                                                     | SurfDock                                  | -                                               |
| **SAbDab**                      | 抗体-抗原复合物（结构抗体数据库）              | - 训练集: 按CDR-H3序列聚类后剩余结构<br>- 测试集: 19个复合物（5个聚类）                 | Antigen-Specific Antibody Design          | http://opig.stats.ox.ac.uk/webapps/sabdab/      |
| **PDB（通用）**                     | 蛋白质结构数据（训练/验证）                 | 未明确数量（覆盖广泛类型）                                                  | RFDiffusion, AlphaFold 3, SE(3) Diffusion | https://www.rcsb.org/                           |
| ~~**Benchmark数据集**~~            | ~~功能位点支架设计评估~~                 | ~~25个设计问题~~                                                    | ~~RFDiffusion~~                           | ~~-~~                                           |
| **CrossDocked**                 | 蛋白质-小分子复合物（过滤后）                | - 训练: 100k<br>- 测试: 100                                        | PocketGen, Generalized Pocket Generation  | https://github.com/gnina/CrossDocked2020        |
| **Binding MOAD**                | 实验验证的蛋白质-小分子复合物                | - 训练: 40k<br>- 测试: 100                                         | PocketGen, Generalized Pocket Generation  | http://www.bindingmoad.org/                     |
| **PPDBench**                    | **非冗余蛋白质-肽复合物**<br><br><br>    | 133个                                                           | Generalized Pocket Generation             | -                                               |
| **PDDBind RNA**                 | 蛋白质-RNA复合物                     | 56个（RNA长度5-15）                                                 | Generalized Pocket Generation             | -                                               |
| **UniRef90等**                   | 多序列比对（MSA）和训练数据增强              | 未明确数量                                                          | AlphaFold 3                               | https://www.uniprot.org/                        |
| **CASP15 RNA Targets**          | RNA结构预测评估                      | 10个RNA目标                                                       | AlphaFold 3                               | https://predictioncenter.org/                   |

1. ## **PepGLAD** Full-Atom Peptide Design with Geometric Latent Diffusion

### 数据集

1. **PepBench**
   
   1. **来源**：Protein Data Bank (PDB)
   
   2. **大小**：
      
      - 训练集：4,157 个条目，952 个簇
      
      - 验证集：114 个条目，50 个簇
      
      - 测试集：93 个条目，93 个簇
   
   3. **内容**：包含受体长度大于 30 个残基、配体长度在 4 到 25 个残基之间的非冗余蛋白质-肽复合物。

2. **PepBDB**
   
   1. **来源**：PepBDB
   
   2. **大小**：
      
      - 训练集：8,434 个条目，1,617 个簇
      
      - 验证集：370 个条目，95 个簇
      
      - 测试集：190 个条目，190 个簇
   
   3. **内容**：蛋白质-肽相互作用的结构数据库。

3. **ProtFrag**
   
   1. **来源**：PDB 中的单体蛋白质
   
   2. **大小**：70,645 个片段
   
   3. **内容**：用于训练变分自编码器的无监督数据。

### 评价指标

1. **序列-结构协同设计（Sequence-Structure Co-Design）**
   
   1. **多样性（Diversity）**：通过序列和结构的唯一簇数量衡量。
   
   2. **一致性（Consistency）**：通过序列和结构聚类标签的 Cramer's V 关联衡量。
   
   3. **结合自由能（ΔG）**：使用 Rosetta 计算的结合能量（kcal/mol），越低表示结合越强。
   
   4. **成功率（Success）**：ΔG < 0 的候选比例。

2. **结合构象生成（Binding Conformation Generation）** **median**
   
   1. **RMSD_{Cα}**：Cα 原子的均方根偏差（Å）。
   
   2. **RMSD_{atom}**：所有原子的均方根偏差（Å）。
   
   3. **DockQ**：评估候选与参考复合物界面全原子相似性的综合指标（0 到 1）。

### 实验与结果

#### 1. 序列-结构协同设计

- **比较方法**：
  
  - HSRN
  
  - dyMEAN
  
  - DiffAb
  
  - RFDiffusion
  
  - AnchorExtension

- **结果（PepBench 测试集）**：
  
  |         |           |           |            |            |
  | ------- | --------- | --------- | ---------- | ---------- |
  | 方法      | 多样性 (↑)   | 一致性 (↑)   | ΔG (↓)     | 成功率        |
  | HSRN    | 0.158     | 0.0       | ≥ 0        | 10.46%     |
  | dyMEAN  | 0.150     | 0.0       | -2.26      | 14.60%     |
  | DiffAb  | 0.427     | 0.670     | -21.20     | 49.87%     |
  | PepGLAD | **0.506** | **0.789** | **-21.94** | **55.97%** |

#### 2. 结合构象生成

- **比较方法**：
  
  - FlexPepDock
  
  - AlphaFold 2
  
  - dyMEAN
  
  - HSRN
  
  - DiffAb

- **结果（PepBench 测试集）**：
  
  |             |               |                 |           |
  | ----------- | ------------- | --------------- | --------- |
  | 方法          | RMSD_{Cα} (↓) | RMSD_{atom} (↓) | DockQ (↑) |
  | FlexPepDock | 6.43          | 7.52            | 0.393     |
  | AlphaFold 2 | 8.49          | 9.20            | 0.355     |
  | dyMEAN      | 7.96          | 8.35            | 0.374     |
  | HSRN        | 6.02          | 7.59            | 0.508     |
  | DiffAb      | 4.23          | 7.60            | 0.586     |
  | PepGLAD     | **4.09**      | **5.30**        | **0.592** |

#### 3. 消融实验

- **测试模块**：
  
  - 全原子几何（Full-Atom）
  
  - 仿射变换（Affine）
  
  - 无监督数据（ProtFrag）
  
  - 掩码策略（Mask）

- **结果**：
  
  |        |         |         |        |        |
  | ------ | ------- | ------- | ------ | ------ |
  | 消融模块   | 多样性 (↑) | 一致性 (↑) | ΔG (↓) | 成功率    |
  | 完整模型   | 0.506   | 0.789   | -21.94 | 55.97% |
  | 无全原子几何 | 0.441   | 0.751   | -20.87 | 51.18% |
  | 无仿射变换  | 0.450   | 0.740   | -19.08 | 52.39% |
  | 无无监督数据 | 0.535   | 0.760   | -20.16 | 52.15% |
  | 无掩码策略  | 0.422   | 0.741   | -20.45 | 57.44% |
2. ## **SurfDock** is a surface-informed diffusion generative model for reliable and accurate protein–ligand complex prediction

### 公共数据集

1. **PDBbind2020**
   
   1. **用途**：训练和测试模型
   
   2. **大小**：363个样本（时间分割测试集）
   
   3. **备注**：包含未见过的蛋白质（144个复合物）

2. **Astex Diverse Set**
   
   1. **用途**：评估药物样小分子的性能
   
   2. **大小**：85个案例

3. **PoseBusters Benchmark Set**
   
   1. **用途**：评估生成姿势的合理性和泛化性
   
   2. **大小**：428个案例

4. **DockGen Dataset**
   
   1. **用途**：测试模型在新结合域/口袋上的表现
   
   2. **大小**：189个复合物（DockGen-full），85个复合物（DockGen-cluster）

5. **DEKOIS 2.0**
   
   1. **用途**：虚拟筛选性能评估
   
   2. **大小**：81个蛋白质靶标

6. **ALDH1B1 实验数据**
   
   1. **用途**：实际应用验证
   
   2. **大小**：37,410个内部药物样小分子

### 评价指标

1. **Docking Success Rate**
   
   1. **标准**：RMSD ≤ 2 Å（成功），RMSD ≤ 1 Å（严格标准）
   
   2. **其他**：中位数RMSD（Med(Å)）

2. **Pose Plausibility (PB-valid)**
   
   1. **工具**：PoseBusters
   
   2. **标准**：通过物理和化学合理性测试的姿势比例

3. **Virtual Screening Metrics**
   
   1. **指标**：
      
      - BED_ROC（早期活性化合物识别）
      
      - Enrichment Factors (EF 0.5%, EF 1.0%, EF 5.0%)
      
      - ROC_AUC（整体分类准确性）
      
      - PR_AUC（精确率-召回率权衡）

4. **Generalizability**
   
   1. **标准**：在低序列相似性（<30%）蛋白质上的表现

### 实验设计与结果

#### 1. Docking Performance on PDBbind2020

- **比较方法**：
  
  - **DL方法**：EquiBind, TANKBind, E3Bind, KarmaDock, DiffDock, DiffDock-L
  
  - **传统方法**：Uni-Dock, Glide SP, GNINA, SMINA, Vina

- **结果**：
  
  - SurfDock成功率：68.41%（RMSD ≤ 2 Å），40.99%（RMSD ≤ 1 Å）
  
  - 在未见蛋白质上：60.88%（RMSD ≤ 2 Å）
  
  - 后处理优化后：PB-valid提升19%

#### 2. Performance on Astex Diverse & PoseBusters Sets

- **结果**：
  
  - Astex Diverse Set：93%成功率（RMSD ≤ 2 Å）
  
  - PoseBusters Set：78%成功率（RMSD ≤ 2 Å）
  
  - 低序列相似性蛋白质上表现稳定

#### 3. Ligand Flexibility Analysis

- **分组**：刚性（≤5键）、中等（5-10键）、柔性（>10键）

- **结果**：
  
  - SurfDock在柔性组表现优于传统方法（接近80%成功率）
  
  - 传统方法在柔性组性能下降15-40%

#### 4. Virtual Screening on DEKOIS 2.0

- **比较方法**：KarmaDock, Glide SP, Surflex-dock, LeDock, Gold, Vina, TANKBind

- **结果**：
  
  - SurfDock在EF 0.5%达到21.00，显著优于其他方法

#### 5. Performance on Apo Structures

- **方法**：使用ESMFold预测结构

- **结果**：
  
  - SurfDock成功率：53%（RMSD ≤ 2 Å）
  
  - 后处理优化后PB-valid显著提升

#### 6. Real-world Application: ALDH1B1 Inhibitors

- **实验流程**：虚拟筛选 → 聚类 → 实验验证

- **结果**：
  
  - 发现7个新支架抑制剂，命中率8.3%
  
  - SPR测定的结合亲和力：0.44-10.10 μM
3. ## **RFDiffusion** Broadly applicable and accurate protein design by integrating structure prediction networks and diffusion generative models

### 公共数据集

1. **Protein Data Bank (PDB)**
   
   1. **用途**：用于训练和验证RFdiffusion模型，包含蛋白质结构数据。
   
   2. **规模**：未明确提及具体数量，但覆盖了广泛的蛋白质结构类型（如单体、寡聚体、酶活性位点等）。

2. **Benchmark数据集**
   
   1. **用途**：评估RFdiffusion在功能位点支架设计中的性能。
   
   2. **规模**：包含25个设计问题，涵盖病毒表位、受体陷阱、小分子结合位点、酶活性位点等。

### 评价指标

1. **结构预测准确性**
   
   1. **RMSD**：设计结构与AlphaFold2 (AF2) 预测结构之间的均方根偏差（阈值：全局<2 Å，功能位点<1 Å）。
   
   2. **TM Score**：衡量设计结构与天然结构的相似性（阈值>0.5）。
   
   3. **AF2置信度**：预测对齐误差（pAE<5）和pLDDT（>80）。

2. **实验验证指标**
   
   1. **结合亲和力**（如MDM2结合设计：0.5-0.7 nM vs 天然p53肽的600 nM）。
   
   2. **SEC色谱**：验证寡聚体组装状态（95%置信区间内）。
   
   3. **电子显微镜（nsEM/cryo-EM）**：2D分类和3D重建验证对称组装。
   
   4. **圆二色谱（CD）**：验证二级结构与设计模型的一致性。

### 实验设计与结果

#### 1. 无条件蛋白质单体生成

- **数据集**：PDB中的单体结构。

- **比较方法**：Hallucination（基于RosettaTTAFold）。

- **结果**：
  
  - RFdiffusion生成300-600个残基的蛋白质，AF2预测RMSD<2 Å。
  
  - 成功率显著高于Hallucination（尤其>100残基时）。
  
  - 实验验证：CD光谱和热稳定性与设计模型一致。

#### 2. 对称寡聚体设计

- **对称类型**：环状（C3-C12）、二面体（D2-D4）、二十面体等。

- **比较方法**：无直接对比（此前方法难以生成高阶对称结构）。

- **结果**：
  
  - **SEC验证**：70/608设计符合预期组装状态。
  
  - **nsEM验证**：2D分类和3D重建显示与设计模型高度一致（如HE0902二十面体）。

#### 3. 功能位点支架设计

- **数据集**：25个基准问题（来自6篇文献）。

- **比较方法**：RFjoint Inpainting、Hallucination、其他DDPMs。

- **结果**：
  
  - RFdiffusion解决23/25问题，成功率高于其他方法（如MDM2结合设计成功率55/95）。
  
  - 实验验证：0.5 nM高亲和力结合MDM2。

#### 4. 酶活性位点支架

- **数据集**：来自PDB的酶活性位点（EC1-5类）。

- **结果**：
  
  - AF2预测活性位点RMSD<1.5 Å（如EC2: 1.29 Å）。
  
  - 通过外部势能隐式建模底物口袋。

#### 5. 蛋白质结合剂设计

- **靶点**：流感血凝素（HA）、IL-7Rα、PD-L1、TrkA。

- **比较方法**：Rosetta物理方法。

- **结果**：
  
  - 实验成功率18%（比Rosetta高2个数量级）。
  
  - 高亲和力结合（如IL-7Rα设计：30 nM）。

### 关键指标总结

|        |        |                       |                         |
| ------ | ------ | --------------------- | ----------------------- |
| 任务     | 数据集/规模 | 比较方法                  | 关键指标（RFdiffusion）       |
| 单体生成   | PDB    | Hallucination         | 300残基设计AF2 RMSD<1.2 Å   |
| 对称寡聚体  | PDB    | 无                     | 70/608 SEC验证成功，nsEM结构匹配 |
| 功能位点支架 | 25基准问题 | RFjoint/Hallucination | 23/25问题解决，MDM2结合0.5 nM  |
| 结合剂设计  | 4个靶点   | Rosetta               | 18%实验成功率，IL-7Rα结合30 nM  |

4. ## Accurate structure prediction of biomolecular interactions with **AlphaFold 3**

### 公共数据集

1. **Protein Data Bank (PDB)**
   
   1. 用途：训练和评估模型
   
   2. 版本：训练数据截至2021年9月30日，评估数据为2022年5月1日至2023年1月12日发布的PDB复合物
   
   3. 数据量：8,856个PDB复合物（评估集）

2. **PoseBusters Benchmark**
   
   1. 用途：评估蛋白质-配体相互作用
   
   2. 版本：V1（2023年8月）和V2（2023年11月）
   
   3. 数据量：V1（428个结构），V2（308个结构）

3. **CASP15 RNA Targets**
   
   1. 用途：评估RNA结构预测
   
   2. 数据量：10个公开可用的RNA目标

4. **其他数据库**
   
   1. UniRef90、UniClust30、MGnify clusters、BFD、RFam、RNACentral、JASPAR等，用于生成多序列比对（MSA）和训练数据增强。

### 评价指标

1. **蛋白质-配体相互作用**
   
   1. 指标：口袋对齐的配体RMSD（<2 Å为成功）
   
   2. 其他指标：PoseBusters有效性检查（PB-valid）

2. **蛋白质-核酸相互作用**
   
   1. 指标：界面LDDT（ILDDT）

3. **RNA结构预测**
   
   1. 指标：RNA LDDT、TM-score、GDT

4. **蛋白质-蛋白质相互作用**
   
   1. 指标：DockQ（>0.23为正确，>0.8为高精度）

5. **共价修饰**
   
   1. 指标：修饰残基的RMSD（<2 Å为成功）

6. **单体蛋白质结构**
   
   1. 指标：LDDT

### 实验与方法比较

1. **蛋白质-配体对接**
   
   1. 比较方法：
      
      - **AF3**：盲对接（仅序列和配体SMILES输入）
      
      - **传统对接工具**：Vina、Gold（使用holo蛋白结构）
      
      - **深度学习工具**：EquiBind、TankBind、DiffDock、Uni-Mol
   
   2. 结果：
      
      - AF3在PoseBusters V1上成功率为76.4%（Vina为52.3%）。
      
      - AF3在PoseBusters V2上成功率为80.5%（Vina为59.7%）。

2. **蛋白质-核酸相互作用**
   
   1. 比较方法：
      
      - **AF3** vs. **RoseTTAFold2NA**
   
   2. 结果：
      
      - AF3在蛋白质-RNA界面ILDDT为64.8，RoseTTAFold2NA为19.0。

3. **RNA结构预测**
   
   1. 比较方法：
      
      - **AF3** vs. **RoseTTAFold2NA**、**Alchemy_RNA2**（人类干预）
   
   2. 结果：
      
      - AF3在CASP15 RNA目标上平均LDDT为47.3，优于RoseTTAFold2NA（35.5）。

4. **共价修饰预测**
   
   1. 结果：
      
      - 键合配体预测成功率为78.5%，糖基化为72.1%（单残基）。

5. **蛋白质-蛋白质相互作用**
   
   1. 比较方法：
      
      - **AF3** vs. **AlphaFold-Multimer v2.3 (AF-M 2.3)**
   
   2. 结果：
      
      - AF3在所有蛋白质-蛋白质界面DockQ>0.23的比例为76.6%（AF-M 2.3为67.5%）。
      
      - 抗体-抗原界面AF3成功率为62.9%（AF-M 2.3为29.6%）。

6. **单体蛋白质结构**
   
   1. 结果：
      
      - AF3的LDDT为86.9，优于AF-M 2.3的85.5。

7. ## Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models for Protein Structures

### **公共数据集**

- **数据集名称**: SAbDab (Structural Antibody Database)

- **数据集大小**:
  
  - 训练集: 从SAbDab中去除分辨率低于4Å和非蛋白质抗原的抗体，并按CDR-H3序列在50%序列同一性下聚类后，剩余的结构。
  
  - 测试集: 手动选择的5个聚类，包含19个抗体-抗原复合物，抗原来自SARS-CoV-2、MERS、流感等病原体。

### **评价指标**

1. **IMP (Improved Binding Energy Percentage)**: 设计的CDR结合能（ΔG）低于原始CDR的百分比。

2. **RMSD (Root-Mean-Square Deviation)**: 生成的CDR结构与原始结构的Cα原子均方根偏差（仅对齐抗体框架）。

3. **AAR (Amino Acid Recovery Rate)**: 生成的CDR序列与参考序列的氨基酸同一性百分比。

### **实验设计与结果**

#### **1. 序列-结构协同设计 (Sequence-Structure Co-design)**

- **比较方法**:
  
  - **DiffAb** (本文提出的扩散模型)
  
  - **RosettaAntibodyDesign (RAbD)** (基于Rosetta能量函数的传统方法)

- **实验结果**:
  
  - **AAR**: DiffAb在所有CDR区域（H1, H2, H3, L1, L2, L3）的表现优于RAbD（例如H1: 65.75% vs. 22.85%）。
  
  - **RMSD**: DiffAb生成的CDR-H3结构多样性更高（RMSD: 3.597Å vs. 2.900Å）。
  
  - **IMP**: DiffAb在CDR-H3上表现与RAbD相当（23.63% vs. 23.25%），但在其他CDR区域略低。

#### **2. 固定骨架的序列设计 (Fix-Backbone Sequence Design)**

- **比较方法**:
  
  - **DiffAb**
  
  - **FixBB** (基于Rosetta的固定骨架序列设计工具)

- **实验结果**:
  
  - **AAR**: DiffAb在所有CDR区域显著优于FixBB（例如H1: 87.83% vs. 37.14%）。

#### **3. 抗体优化 (Antibody Optimization)**

- **实验设计**: 通过扩散模型对现有抗体的CDR-H3进行扰动和去噪优化。

- **实验结果**:
  
  - 当优化步数 \( t = 4 \) 时，优化后的CDR-H3在保持结构相似性（RMSD: 1.290Å）的同时，结合能改善比例（IMP: 23.29%）接近从头设计的结果。

#### **4. 无结合抗体框架的设计 (Design Without Bound Antibody Frameworks)**

- **实验设计**: 使用HDOCK将抗体模板与抗原对接后，设计CDR-H3。

- **实验结果**: 生成的抗体结合能分布合理，但缺乏实验验证。
6. ## FoldToken4: Consistent & Hierarchical Fold Language

### 公共数据集

1. **PDB数据集**
   
   1. **用途**：训练和评估模型
   
   2. **版本**：截至2024年3月1日的PDB数据
   
   3. **训练集**：162K蛋白质（过滤缺失坐标和长度小于30的蛋白质）
   
   4. **评估集**：
      
      - **T493**：493个蛋白质
      
      - **T116**：116个蛋白质
      
      - **N128**：128个新蛋白质（AlphaFold3发布后）
      
      - **M1031**：多链蛋白质复合物

### 评价指标

1. **结构重建质量**
   
   1. **TMscore**：衡量预测结构与真实结构的相似性（0-1，越高越好）
   
   2. **RMSD**：均方根偏差（Å，越低越好）
   
   3. **对齐RMSD**：使用PyMol计算的RMSD

2. **代码多样性**
   
   1. **相似性指标（Sim）**：衡量代码向量的区分度（越低越好）

3. **多尺度一致性**
   
   1. **过渡矩阵（Transition Matrix）**：分析不同尺度代码间的映射关系

### 实验与方法比较

#### 1. 单链蛋白质重建（Q1）

- **比较方法**：
  
  - **FoldToken系列**：FT1（65536代码）、FT2（65536代码）、FT3（256-4096代码）、FT4（32-4096代码）
  
  - **ESM3**（4096代码）

- **结果**：
  
  - **T116数据集**：
    
    - FT4（4096代码）TMscore=0.95，RMSD=0.67
    
    - FT3（4096代码）TMscore=0.95，RMSD=0.64
  
  - **N128数据集**：
    
    - FT4（32代码）TMscore=0.78，RMSD=2.91
    
    - FT4（4096代码）TMscore=0.91，RMSD=1.34

#### 2. 多链蛋白质复合物重建（Q2）

- **比较方法**：FT2、FT3、FT4

- **结果**：
  
  - **M1031数据集**：
    
    - FT4（4096代码）TMscore=0.93，RMSD=1.34
    
    - FT3（4096代码）TMscore=0.94，RMSD=1.10
  
  - **小代码本表现**：
    
    - FT4（256代码）仍能有效重建复合物（TMscore=0.91，RMSD=1.53）

#### 3. 代码分析与多尺度一致性（Q3）

- **代码多样性**：
  
  - FT4（32代码）Sim=0.7098（优于FT1的0.9959）

- **多尺度一致性**：
  
  - 通过过渡矩阵实现跨尺度代码转换（如从2^12到2^5），无需重新运行模型。
  
  - 示例：图7展示了从精细尺度（4096代码）到粗尺度（32代码）的结构重建。
7. ## **PepFlow** Full-Atom Peptide Design based on Multi-modal Flow Matching

### 数据集

- **来源**：PepBDB (Wen et al., 2019) 和 Q-BioLip (Wei et al., 2024)。

- **处理**：
  
  - 去除重复条目。
  
  - 筛选标准：分辨率 <4Å，肽长度在3-25个残基之间。
  
  - 使用mmseqs2按40%的肽序列相似性聚类。

- **最终数据**：
  
  - 8,365个非孤立复合体，分布在292个聚类中。
  
  - 测试集：随机选择10个聚类，包含158个复合体。
  
  - 训练和验证集：其余复合体。

### 评价指标

#### 1. **Sequence-Structure Co-Design（序列-结构协同设计）**

- **Geometry（几何结构）**：
  
  - **AAR（Amino Acid Recovery）**：生成的肽与天然肽的序列一致性。
  
  - **RMSD（Root-Mean-Square Deviation）**：生成肽与天然肽的Cα原子均方根偏差。
  
  - **SSR（Secondary-Structure Similarity Ratio）**：二级结构相似性比例。
  
  - **BSR（Binding Site Ratio）**：生成肽与天然肽结合位点的重叠比例。

- **Energy（能量）**：
  
  - **Stability**：设计肽的复合体能量低于天然肽的比例。
  
  - **Affinity**：设计肽的结合亲和力高于天然肽的比例。

- **Design（设计质量）**：
  
  - **Designability**：生成序列能折叠成与生成结构相似的肽的比例。
  
  - **Diversity**：生成肽结构的平均结构差异性（1 - TM-Score）。

#### 2. **Fix-Backbone Sequence Design（固定骨架序列设计）**

- **AAR**：序列恢复率。

- **Worst**：最低恢复率。

- **Likeness**：生成序列与天然序列分布的负对数似然。

- **Diversity**：生成序列的平均汉明距离。

- **Designability**：生成序列能折叠成与天然结构相似的肽的比例。

#### 3. **Side-Chain Packing（侧链包装）**

- **MAE（Mean Absolute Error）**：预测侧链角度的平均绝对误差。

- **Correct**：预测角度与真实角度偏差在20°以内的比例。

### 实验与方法比较

#### 1. **Sequence-Structure Co-Design**

- **比较方法**：
  
  - **RFDiffusion**：生成蛋白质骨架，用ProteinMPNN预测序列。
  
  - **ProteinGenerator**：联合生成骨架和序列。
  
  - **Diffusion**：基于扩散的肽生成模型。
  
  - **PepFlow**（三种变体）：
    
    - **w/Bb**：仅生成骨架。
    
    - **w/Bb+Seq**：联合生成骨架和序列。
    
    - **w/Bb+Seq+Ang**：全原子生成（骨架、序列、侧链）。

- **结果分析**：
  
  - PepFlow在AAR、RMSD、SSR、BSR等指标上优于基线方法。
  
  - 全原子版本（w/Bb+Seq+Ang）在结合位点识别（BSR）和亲和力（Affinity）上表现最佳。
  
  - 基线方法在稳定性和设计多样性上略优，但PepFlow在结构准确性上更突出。

#### 2. **Fix-Backbone Sequence Design**

- **比较方法**：
  
  - **ProteinMPNN**：基于GNN的逆折叠模型。
  
  - **ESM-IF**：基于GVP-Transformer的逆折叠模型。
  
  - **PepFlow**（两种变体）：
    
    - **w/Bb+Seq**：仅生成序列。
    
    - **w/Bb+Seq+Ang**：生成序列和侧链。

- **结果分析**：
  
  - PepFlow在序列恢复率（AAR）和多样性上优于基线方法。
  
  - 引入侧链建模后，序列恢复率略有下降，但Likeness更接近天然分布。

#### 3. **Side-Chain Packing**

- **比较方法**：
  
  - **RosettaPacker**：基于能量的方法。
  
  - **SCWRL4**：基于统计能量函数的方法。
  
  - **DLPacker**：基于3D CNN的模型。
  
  - **AttnPacker**：基于注意力机制的模型。
  
  - **DiffPack**：基于扩散的模型。
  
  - **PepFlow w/Bb+Seq+Ang**：全原子生成。

- **结果分析**：
  
  - PepFlow在MAE和Correct比例上表现最佳，尤其在χ1和χ2角度的预测上更准确。

### 实验结果总结

|                        |                          |                                                   |
| ---------------------- | ------------------------ | ------------------------------------------------- |
| 任务                     | 方法                       | 关键指标（表现最佳标粗）                                      |
| **Sequence-Structure** | RFDiffusion              | Designability: 78.52%                             |
| <br>                   | ProteinGenerator         | AAR: 45.82%                                       |
| <br>                   | **PepFlow w/Bb+Seq+Ang** | **AAR: 51.25%**, **RMSD: 2.07Å**, **BSR: 86.89%** |
| **Fix-Backbone**       | ProteinMPNN              | AAR: 53.28%                                       |
| <br>                   | **PepFlow w/Bb+Seq**     | **AAR: 56.40%**, Diversity: 23.38                 |
| **Side-Chain Packing** | **PepFlow w/Bb+Seq+Ang** | **MAE (χ1): 17.38°**, **Correct: 62.79%**         |

8. ## SE(3) diffusion model with application to protein backbone generation

### 数据集

- **Protein Data Bank (PDB)**
  
  - **描述**: 用于蛋白质结构预测和设计的公共数据库。
  
  - **大小**: 训练集包含20,312个蛋白质单体（过滤后）。
  
  - **过滤条件**: 长度在60到512个氨基酸之间，分辨率小于5Å，且二级结构中环（loop）的比例不超过50%。

### 评价指标

1. **设计性 (Designability)**
   
   1. **scTM > 0.5**: TM-score（模板建模分数）大于0.5，表示生成的蛋白质骨架与预测结构具有较高的相似性。
   
   2. **scRMSD < 2Å**: Cα原子的均方根偏差小于2Å，更严格的设计性标准。

2. **多样性 (Diversity)**
   
   1. 使用MaxCluster工具，以TM-score阈值为0.5或0.6对生成的蛋白质骨架进行层次聚类，计算唯一簇的比例。

3. **新颖性 (Novelty)**
   
   1. **pdbTM**: 使用FoldSeek工具搜索生成的蛋白质结构与PDB中已知结构的最高TM-score，值越低表示新颖性越高。

4. **生成速度**
   
   1. 生成100个氨基酸长度的蛋白质骨架所需时间（秒）。

### 实验内容

1. **无条件单体蛋白质骨架生成**
   
   1. **目标**: 评估FrameDiff在生成设计性、多样性和新颖性蛋白质骨架方面的能力。
   
   2. **方法**: 从噪声开始采样蛋白质骨架，使用ProteinMPNN设计序列，并通过ESMFold预测结构。
   
   3. **指标**: scTM、scRMSD、pdbTM、多样性比例。

2. **消融实验**
   
   1. **内容**: 测试不同噪声尺度（$$\zeta$$）、采样步数（$$N_{\text{steps}$$）、序列设计数量（$$N_{\text{seq}$$）对性能的影响。
   
   2. **关键结果**:
      
      - 降低噪声尺度（$$\zeta=0.$$）显著提高设计性（scTM > 0.5达到75%）。
      
      - 增加序列设计数量（$$N_{\text{seq}}=10$$）进一步提升设计性（scTM > 0.5达到84%）。

3. **与其他方法的比较**
   
   1. **对比方法**:
      
      - **Chroma**: 基于非各向同性扩散的蛋白质骨架生成方法。
      
      - **RFdiffusion**: 基于预训练结构的扩散模型。
      
      - **FoldingDiff**: 公开可用的蛋白质生成扩散模型。
   
   2. **结果分析**:
      
      - FrameDiff在无预训练模型中表现最佳（scTM > 0.5达到75%），仅次于RFdiffusion（需预训练）。
      
      - 生成速度比RFdiffusion快34倍（4.4秒 vs. 150秒）。

4. **二级结构分析**
   
   1. **内容**: 分析生成蛋白质的二级结构组成（螺旋、折叠、环）。
   
   2. **结果**: 生成了多种二级结构，长蛋白质（>400氨基酸）更倾向于螺旋结构。

### 实验结果总结

|                       |                     |            |                 |
| --------------------- | ------------------- | ---------- | --------------- |
| **指标**                | **FrameDiff**       | **Chroma** | **RFdiffusion** |
| **设计性 (scTM > 0.5)**  | 75% (\(\zeta=0.1\)) | 55%        | 更高（需预训练）        |
| **设计性 (scRMSD < 2Å)** | 28% (\(\zeta=0.1\)) | 未报告        | 更高              |
| **多样性 (比例)**          | >0.5 (TM=0.6)       | 未报告        | 类似              |
| **生成速度 (100aa)**      | 4.4秒                | 未报告        | 150秒            |
| **新颖性 (pdbTM < 0.6)** | 部分样本达到              | 未报告        | 未报告             |

9. ## Efficient generation of protein pockets with PocketGen

### 公共数据集

1. **CrossDocked数据集**
   
   1. **大小**: 22.5M蛋白质-分子对（过滤后约180k数据点）
   
   2. **用途**: 训练、验证和测试集按30%序列相似性阈值划分
   
   3. **特点**: 通过交叉对接生成，保留结合姿态RMSD ≤ 1Å的样本

2. **Binding MOAD数据集**
   
   1. **大小**: 约41k实验确定的蛋白质-配体复合物（过滤后保留QED ≥ 0.3的配体）
   
   2. **用途**: 按酶分类号（EC编号）划分训练、验证和测试集
   
   3. **特点**: 仅包含标准氨基酸和小分子配体（原子类型限制为C, N, O, S等）

### 评价指标

#### 结合亲和力

- **AutoDock Vina Score** (↓): 结合自由能（越低越好）

- **MM-GBSA** (↓): 结合自由能计算（分子力学/广义玻恩表面积）

- **GlideSP Score** (↓): 基于对接的评分

#### 结构有效性

- **scRMSD** (↓): 生成结构与预测结构的骨架原子RMSD（<1Å为可设计）

- **scTM Score** (↑): 模板建模分数（0-1，越高越好）

- **pLDDT** (↑): 局部结构置信度（0-100，越高越好）

#### 序列-结构一致性

- **AAR (Amino Acid Recovery Rate)** (↑): 正确预测的残基类型百分比

- **Success Rate** (↑): 生成口袋结合亲和力优于参考结构的比例

### 对比方法

|                 |            |                             |
| --------------- | ---------- | --------------------------- |
| 方法名称            | 类型         | 特点                          |
| PocketOptimizer | 物理建模       | 基于能量函数优化突变                  |
| DEPACT          | 模板匹配       | 通过数据库搜索和残基移植设计口袋            |
| RFdiffusion     | 深度学习（扩散模型） | 基于RoseTTAFold的蛋白质生成，后处理序列设计 |
| RFAA            | 深度学习（扩散模型） | RFdiffusion的改进版，直接生成配体结合蛋白  |
| FAIR            | 深度学习（迭代优化） | 两阶段全原子口袋设计                  |
| dyMEAN          | 深度学习（图网络）  | 动态多通道等变图网络，适配抗体设计           |

### 实验结果

#### 主要性能对比（CrossDocked数据集）

|                   |           |             |         |
| ----------------- | --------- | ----------- | ------- |
| 指标                | PocketGen | RFAA (最佳基线) | 提升幅度    |
| AAR               | 63.40%    | ~49.45%     | +13.95% |
| Vina Score (Top1) | -9.655    | -9.216      | -0.439  |
| Success Rate      | 97%       | 93%         | +4%     |
| 生成时间 (100个)       | 44.2秒     | 2,210.1秒    | 50倍加速   |

#### 其他关键结果

1. **口袋大小影响**
   
   1. 设计半径从3.5Å增至5.5Å时，AAR略有下降，但结合亲和力提升（最低Vina达-17.5 kcal/mol）。

2. **PLM规模效应**
   
   1. 使用ESM-2 15B模型时，AAR从35M模型的54.58%提升至66.61%，符合对数缩放规律。

3. **配体特性分析**
   
   1. 大配体（原子数多）倾向于生成更高亲和力的口袋（Pearson ρ=-0.61）。
   
   2. 关键功能基团（如氢键供体、芳香环）显著提升结合亲和力。

4. **案例研究**
   
   1. **HCY/APX/7V7配体**: PocketGen成功保留原始相互作用并新增氢键/π-阳离子相互作用。
   
   2. **未训练蛋白质（PIB/luxsit)**: 在rucaparib和DTZ配体上仍优于基线（Vina Score提升显著）。

5. ## Generalized Protein Pocket Generation with Prior-Informed Flow Matching

### 公共数据集

|              |                         |              |             |
| ------------ | ----------------------- | ------------ | ----------- |
| 数据集名称        | 描述                      | 规模           | 任务类型        |
| CrossDocked  | 跨对接生成的蛋白-小分子复合物（过滤后）    | 100k训练；100测试 | 小分子结合口袋设计   |
| Binding MOAD | 实验验证的蛋白-小分子复合物（按EC编号划分） | 40k训练；100测试  | 小分子结合口袋设计   |
| **PPDBench** | **非冗余蛋白-肽复合物**          | **133**      | **肽结合口袋设计** |
| PDDBind RNA  | 蛋白-RNA复合物（RNA长度5-15）    | 56           | RNA结合口袋设计   |

### 评价指标

- **序列与结构**
  
  - `AAR`: 氨基酸恢复率（%）
  
  - `scRMSD`: 自洽主链RMSD（Å）

- **结合能力**
  
  - `Vina Score`: 小分子结合评分（kcal/mol，越低越好）
  
  - `ΔΔG`: 肽/RNA结合自由能变化（kcal/mol）

- **相互作用**
  
  - `Clash`: 原子冲突数
  
  - `HB`: 氢键数量

### 实验结果

#### 小分子结合口袋（CrossDocked）

|               |           |              |              |
| ------------- | --------- | ------------ | ------------ |
| 方法            | AAR (%) ↑ | Vina Score ↓ | scRMSD (Å) ↓ |
| PocketFlow    | **52.19** | **-8.236**   | 0.67         |
| RFDiffusionAA | 50.85     | -7.012       | 0.68         |
| FAIR          | 40.16     | -7.015       | 0.75         |

#### 跨模态泛化性能

|              |            |                  |
| ------------ | ---------- | ---------------- |
| 数据集          | 方法         | ΔΔG (kcal/mol) ↓ |
| PPDBench (肽) | PocketFlow | **-1.06**        |
| PDDBind RNA  | PocketFlow | **-0.78**        |

# 补充实验

## 目前存在的问题

实验数据集不充分，训练集和测试集有重叠？

baseline不清晰到底和哪些方法对比

消融实验？

## 补充数据集

见表一加粗

## Baseline怎么定

现有的实验方法是follow的pepglad, 然而pepflow非常重量级

需要新增的baseline: **Pepflow**, AlphaFlod3, RFDiffusion

## 新增什么指标

参考Pepflow吧
