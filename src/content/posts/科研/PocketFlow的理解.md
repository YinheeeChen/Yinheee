```yaml
title: PocketFlow的一些理解

published: 2025-04-01

description: ''

image: ''

tags: [Pocket-peptide, Diffusion, Flow matching]

category: '科研'

draft: false

lang: ''
```

这一篇也是审稿人提到的，和flow matching相关的一篇工作，名为pocketflow。他的目的是基于蛋白质-配体相互作用的先验知识，生成高质量的蛋白质口袋，动机在于传统的方法在计算上非常消耗时间，且生成的口袋质量比较低，而深度学习方法虽然提高了效率，但是忽略了蛋白质-配体相互作用的知识。因此他们就提出PocketFlow, 一个基于flow matching的生成模型。

方法上其实就是利用了flow mathching并结合领域知识来生成高质量的蛋白质口袋

![](https://lightwheel.feishu.cn/space/api/box/stream/download/asynccode/?code=MjQwYjg1Nzg0NjQ1YzY4MWU1NjI0YTU2YjUxZWU1NTBfZ0JlV005aEgwWEllSXVGbGtsWGtoUklMV2E1Uld3T0lfVG9rZW46RUlKa2JaT3VVb3hIMEl4bWdJbmNNSDN1bmFnXzE3NDM1NzgwODM6MTc0MzU4MTY4M19WNA)

再继续往下看，可以看到PocektFlow和PepFlow的共同点：在 SE(3) 空间中建模蛋白质骨架的生成，在环面上建模侧链二面角，对比两篇工作可以发现完全一样。。。奇怪的是这两篇并没有相互引用:(

而pocketflow有一个是分类残基类型和相互作用类型的方法：每个残基被表示为一个20维的概率向量，对应20种氨基酸类型。选择均匀分布作为残基类型的先验分布，接着使用CFM来预测分布，并使用交叉熵损失来计算损失。而相互作用类型的分类方法和分类残基类似，只是把他们建模成分类的数据。

网络架构基于修改版的 FrameDiff，结合了 AlphaFold2 中的 Invariant Point Attention（IPA）和 transformer 层，以有效捕获蛋白质和配体的三维结构和序列特征。

他的网络架构其实并不是非常创新，都是基于已有工作的，并未找到图，且他说得比较乱：

**特征初始化**

- **节点嵌入**：使用多层感知机（MLP）和正弦嵌入对残基索引和时间步进行编码，初始化节点嵌入。

- **边嵌入**：除了节点嵌入，边嵌入还考虑了相对序列距离和预测的 Cα 位移，通过 MLP 进行编码。

**节点更新**

- **Invariant Point Attention (IPA)**：来自 AlphaFold2，用于更新节点嵌入，捕获残基之间的空间关系。

- **Transformer 层**：标准的 Transformer 架构用于更新节点嵌入，捕获序列级别的特征。

- **残差连接**：通过跳跃连接（skip connections）保留原始信息，增强模型的训练稳定性。

**边更新**

- **多层感知机 (MLP)**：用于更新边嵌入，结合当前边嵌入和源、目标节点嵌入。

**骨架更新**

- **帧更新算法**：基于 AlphaFold2 的算法，通过线性变换更新残基的骨架框架，包括旋转和平移。

**残基/相互作用类型和二面角预测**

- **多层感知机 (MLP)**：用于预测残基类型、相互作用类型和侧链二面角。

- **Softmax 层**：将预测结果转换为概率分布，用于分类任务。

其实这个工作要是根据PepFlow的说法也可以称得上是多模态，他的网络架构也就是根据多个模态进行分别设计的，然后每个模态都会用到Flow Matching来进行生成。。。

说实话这个有点难懂，需要拆开理解一下：

这里用流匹配进行了四个方面的建模：SE(3)，环面，残基，相互作用，也就是说有4个loss function，训练的时候把他们求和
