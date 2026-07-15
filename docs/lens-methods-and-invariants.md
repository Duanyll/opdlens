# Lens 方法、跨模型匹配与不变量分析

本文整理当前 `opdlens` 中 B、C、D、E 四个实验臂的设计，并重点讨论：

1. logit lens 和 Jacobian lens 如何从中间层提取信息；
2. B、C、E 如何跨模型匹配这些信息；
3. 当前 Jacobian lens 校准语料与 pretrain-like 语料的差异；
4. 为什么较大的 lens auxiliary loss 会导致训练坍塌；
5. Training-Free Looped Transformers 暗示了什么不变量；
6. OPRD-Bridge 的原始 recipe 与当前 Arm D 有什么区别；
7. 下一轮最有区分度的实验是什么。

## 1. 核心判断

Training-Free Looped Transformers 和 lens 蒸馏真正相交的地方，不是它们使用了相同的 workspace 定义，而是它们都碰到了同一个问题：

> 中间表示必须处在下游层所期待的计算阶段，并且整个深度方向上的演化必须落在原模型训练过的终点附近。

Looped Transformer 论文并没有直接用 J-lens 识别或抽出 workspace。它循环的是完整的中间 residual block。把这个中间窗口解释成 workspace 是一个合理假设，但不是论文已经证明的结论。

这篇论文真正提供的关键思想是：重复一个 residual update 时，必须保持总积分时间不变。否则，后续层会收到一个位于错误计算阶段的 hidden state。

对于当前 B、C、E，最可能被破坏的主不变量是：

> **Depth-time / endpoint interface invariant：匹配的表示必须对应相同的计算时间和 causal horizon，而且 auxiliary update 不能把 student 的中间状态推离下游层训练过的输入流形。**

除此之外，还有两个次级不变量：

- lens score 的 scale、entropy 和 vocabulary tail 没有经过跨模型校准；
- offline lens 是在 frozen model 上拟合的，但训练会改变 hidden、final norm 和 LM head，使测量接口本身失效。

因此，当前现象更像 objective/geometric instability，而不是浮点数值上的 NaN 或 Inf。

## 2. 统一记号和 tensor 形状

记：

| 记号 | 意义 | 形状 |
|---|---|---|
| $T$ | 序列长度 | scalar |
| $d$ | residual stream 宽度 | scalar |
| $V$ | 词表大小 | scalar |
| $H_l$ | 第 $l$ 个 block 输出的 hidden states | $[T,d]$ |
| $N$ | 模型自己的 final norm | $[T,d]\rightarrow[T,d]$ |
| $U$ | 模型自己的 unembedding / LM head | $[V,d]$ |
| $Z_l$ | lens 输出的 vocabulary logits | $[T,V]$ |

当前代码中的 block $l$ 输出对应 Hugging Face 的 `hidden_states[l + 1]`。读取中间层和 unembedding 的实现位于 [`opdlens/models.py`](../opdlens/models.py)。

## 3. Logit lens 如何提取信息

### 3.1 计算过程

第一步，取得第 $l$ 层 hidden：

$$
H_l\in\mathbb{R}^{T\times d}
$$

第二步，应用模型自己的 final norm：

$$
\widehat H_l=N(H_l)
$$

形状不变：

$$
[T,d]\rightarrow[T,d]
$$

第三步，应用模型自己的 LM head：

$$
Z_l^{LL}=\widehat H_lU^\top
$$

输出形状：

$$
[T,d]\rightarrow[T,V]
$$

如果需要概率分布，再计算：

$$
P_l^{LL}=\operatorname{softmax}(Z_l^{LL}/\tau)
$$

输出仍然是：

$$
[T,V]
$$

### 3.2 它的含义

Logit lens 问的是：

> 如果现在立刻把第 $l$ 层 residual 当成最终层 residual，模型的 LM head 会把它读成哪些 token？

它隐含了一个假设：不同深度的 residual stream 使用同一套最终层坐标。

这个假设在最后几层通常比较合理，因为 residual connection 会保持较强的坐标连续性；在较早层则经常不成立，因为表示还会被后续层旋转、重组和路由。

因此，logit lens 更像一个便宜的、偏后期层的解释探针，不是一个保证校准的中间层概率模型。

## 4. Jacobian lens 如何提取信息

### 4.1 完整 Jacobian

第 $l$ 层位置 $i$ 的 hidden perturbation 会影响最终层位置 $j$ 的 hidden。完整导数为：

$$
\frac{\partial h_{L,j}}{\partial h_{l,i}}
\in\mathbb{R}^{d\times d}
$$

如果保留所有 source 和 target positions，完整 tensor 的概念形状是：

$$
\frac{\partial H_L}{\partial H_l}
\in\mathbb{R}^{T_{out}\times d\times T_{in}\times d}
$$

它过于庞大，而且强烈依赖具体 prompt，所以 J-lens 不直接保存这个 tensor。

### 4.2 对 position 和 prompt 做平均

论文定义的核心思想是：对 source position、所有未来 target positions 和一个较大的语料分布做平均：

$$
\overline J_l
=
\mathbb E_{x,i,j\ge i}
\left[
\frac{\partial h_{L,j}}{\partial h_{l,i}}
\right]
$$

最终每层只保存一个矩阵：

$$
\overline J_l\in\mathbb{R}^{d\times d}
$$

它表示：

> 在平均上下文中，第 $l$ 层的某个 residual direction 会怎样影响当前位置或未来位置的最终 residual direction？

当前 [`opdlens/jlens/fitting.py`](../opdlens/jlens/fitting.py) 的具体 estimator 是：

$$
\overline J_l^{prompt}
=
\frac{1}{|S|}
\sum_{i\in S}
\sum_{j\in S,j\ge i}
\frac{\partial h_{L,j}}{\partial h_{l,i}}
$$

然后再对 prompts 做平均。

实现上，它在所有有效 target positions 的同一个 output dimension 上同时放置 one-hot cotangent，反向传播一次得到对应的 Jacobian row。遍历所有 output dimensions 后得到完整的 $[d,d]$ 矩阵。

这里有一个需要单独验证的细节：当前实现是对 future targets 求和、对 source positions 求平均。这样 prompt 长度和 causal horizon 会改变不同导数方向在最终矩阵中的权重。官方论文的文字描述更接近对 source、target、prompt 全部做 expectation。

### 4.3 应用 Jacobian lens

第一步，取得中间 hidden：

$$
H_l\in\mathbb{R}^{T\times d}
$$

第二步，把它运输到最终层 residual 坐标：

$$
\widetilde H_l^J
=
H_l\overline J_l^\top
$$

形状为：

$$
[T,d]\times[d,d]\rightarrow[T,d]
$$

第三步，使用模型自己的 final norm 和 LM head：

$$
Z_l^{JL}
=
N(\widetilde H_l^J)U^\top
$$

输出形状为：

$$
[T,d]\rightarrow[T,V]
$$

最后可计算：

$$
P_l^{JL}=\operatorname{softmax}(Z_l^{JL}/\tau)
$$

### 4.4 它和 logit lens 的区别

Logit lens 相当于令：

$$
\overline J_l=I
$$

Jacobian lens 则使用实际平均 downstream derivative，校正不同深度之间的表示旋转。

这使 J-lens 更容易在早期和中间层发现可 verbalize 的 concept direction，也是 Anthropic 将它与 global workspace 联系起来的原因。原始方法见 [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens)。

### 4.5 它不是什么

严格的一阶 Taylor approximation 应当包含 reference point 和 intercept：

$$
h_L(h_l)
\approx
h_L(h_l^0)
+
J_l(h_l-h_l^0)
$$

当前 lens 只使用：

$$
J_lh_l
$$

因此它不是一个严格校准的最终 hidden predictor。它更接近一个平均 causal direction probe。

原始 J-lens 工作主要使用 top tokens、rank、cosine similarity 和 sparse concept decomposition 来解释结果。它没有证明不同模型的完整 softmax 分布是可直接匹配的。

## 5. 不同模型的 lens 信息凭什么可以匹配

跨模型匹配并不因为输出都叫 vocabulary logits 就自动成立。至少需要满足以下条件：

1. 两侧 readout 测量的是相同的语义 observable；
2. 两侧表示处在相同的计算阶段；
3. 两侧聚合了相同的 causal horizon；
4. 两侧 logits 的 scale、entropy 和 tail 已经校准；
5. 离线 lens 在当前训练分布和当前 checkpoint 上仍然有效；
6. 优化这个 readout 不会反过来破坏 readout 的测量意义。

可以定义两个 readout：

$$
R_m^I(h)=U_mN_m(h)
$$

$$
R_m^J(h)=U_mN_m(J_mh)
$$

其中 $m$ 表示 student 或 teacher。

即使 student 和 teacher 的最终 token labels 相同， $R_s$ 和 $R_t$ 仍可能具有不同的 logit scale、entropy、极值结构和长尾噪声。

## 6. B、C、E 当前如何匹配

### 6.1 层映射

当前代码先把 teacher layer 按深度比例映射到 student layer：

$$
l_s
=
\operatorname{round}
\left(
\frac{(l_t+1)N_s}{N_t}
\right)-1
$$

然后限制在 student 的有效中间层范围内。

对于当前 32-layer teacher 和 24-layer student，teacher 的层 $8$ 、 $16$ 、 $24$ 大致对应 student 的层 $6$ 、 $12$ 、 $18$ 。

这个映射只对齐 normalized depth，不保证两个模型在该深度完成了相同的计算阶段。

### 6.2 通用 lens loss

从 completion positions 中选择最多 $n$ 个位置后，每侧 readout 的形状为：

$$
Z_s,Z_t\in\mathbb{R}^{n\times V}
$$

默认 forward KL 为：

$$
L_{lens}
=
\frac{1}{K}
\sum_{k=1}^{K}
D_{KL}
\left(
\operatorname{softmax}(Z_t^k/\tau)
\;\|\;
\operatorname{softmax}(Z_s^k/\tau)
\right)
$$

最后得到一个 scalar。当前也支持 reverse KL 和 teacher top-k vocabulary filtering。共享实现位于 [`opdlens/losses.py`](../opdlens/losses.py)。

### 6.3 Arm B：双侧 logit lens

Student：

$$
Z_s=R_s^I(h_s)
$$

Teacher：

$$
Z_t=R_t^I(h_t)
$$

隐含假设：比例深度处的两个即时 logit-lens readout 表示相同的计算阶段。

优点是左右形式对称，不依赖离线 Jacobian。

问题是：

- normalized depth 不等于 semantic time；
- early logit lens 本身可能没有意义；
- 两侧的概率 calibration 没有保证；
- dense KL 会匹配大量不可靠 vocabulary tail。

### 6.4 Arm C：student logit lens 对 teacher Jacobian lens

Student：

$$
Z_s=R_s^I(h_s)
$$

Teacher：

$$
Z_t=R_t^J(h_t)
$$

这是 causal horizon 最不对称的设计。

Teacher J-lens 聚合了该层表示对当前位置和未来最终层表示的平均影响，student 却使用即时 logit lens。强 supervision 可能迫使 student 在中间层提前产生 terminal/verbalizable state。

随后 student 的剩余层仍会继续执行完整 residual computation，因此容易出现 semantic-time overshoot。

### 6.5 Arm E：双侧 Jacobian lens

Student：

$$
Z_s=R_s^J(h_s)
$$

Teacher：

$$
Z_t=R_t^J(h_t)
$$

它在形式上最对称：两侧都被运输到各自的 final-residual coordinates。

但是 student Jacobian 是在初始化模型上离线拟合的：

$$
J_s=J_s^{step\ 0}
$$

训练时，student hidden、attention、MLP、final norm 和 LM head 会不断变化。于是固定的 $J_s$ 会逐渐离开它原来的线性化有效域。

这会产生 probe hacking：student 可以降低 lens loss，却不再保持 lens 原来测量的真实因果属性。

## 7. 当前校准语料与 pretrain-like 分布的差异

### 7.1 比较设置

使用当前 Qwen3.5 tokenizer，对以下语料做统计：

- GSM8K：前 264 条 question 和 gold answer，包含 chat formatting，截断到 384 tokens；
- MATH：264 条、七个 subject 交错的 problem 和 full solution，截断到 384 tokens；
- FineWeb proxy：前 1000 篇文档，每篇取前 128 tokens。

FineWeb 只是英文 web pretrain-like proxy，不代表 Qwen 实际的多语言、代码和数学预训练混合。数据源见 [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)。

### 7.2 Unigram 统计

| 校准集 | tokens | unique token types | entropy | effective vocab | top-100 mass |
|---|---:|---:|---:|---:|---:|
| GSM8K CoT | 53,699 | 2,677 | 7.62 bits | 196 | 75.5% |
| MATH CoT | 69,453 | 2,625 | 8.03 bits | 262 | 71.1% |
| FineWeb proxy | 124,481 | 18,922 | 10.75 bits | 1,726 | 45.1% |

### 7.3 分布距离

| Pair | JSD，范围为 0 到 1 bit | Total variation | 相同 token budget 下的 unique types |
|---|---:|---:|---:|
| GSM8K vs FineWeb | 0.442 | 0.615 | 2,677 vs 11,891 |
| MATH vs FineWeb | 0.522 | 0.663 | 2,625 vs 13,797 |

Total variation 为 0.615 或 0.663，意味着大约 62% 或 66% 的 unigram probability mass 需要被移动，两个分布才能相同。这是很大的 domain shift。

去掉 chat template 后：

- GSM8K 对 FineWeb 的 JSD 从 0.442 只降到 0.431；
- MATH 对 FineWeb 的 JSD 从 0.522 只降到 0.516。

所以主要差异来自内容，而不是 chat template。

GSM8K 被数字、单位、`<<`、`>>`、`=` 和 `####` 等结构强烈支配。MATH 被 LaTeX、括号、`frac`、`cdot` 和指数结构强烈支配。

### 7.4 应不应该改用 pretrain-like 语料

如果目标是拟合一个通用 global-workspace lens，那么答案是应该以 pretrain-like 为主。Anthropic 原始方法是在一千个 pretraining-like prompts 上平均 Jacobian。

但如果目标是 on-policy 数学蒸馏，纯 pretrain-like 也不够。需要区分三个 estimand：

$$
J_l^{broad}
=
\mathbb E_{x\sim D_{pretrain}}
[J_l(x)]
$$

$$
J_l^{task}
=
\mathbb E_{x\sim D_{gold\ task}}
[J_l(x)]
$$

$$
J_l^{onpolicy}
=
\mathbb E_{x\sim\pi_s}
[J_l(x)]
$$

它们回答的是不同问题：

- broad lens 测量一般上下文中的 verbalizable directions；
- task lens 测量数学 CoT 条件下的 directions；
- on-policy lens 测量 student 实际会访问的状态附近的 directions。

更广的语料可以覆盖更多 residual dynamics、attention patterns 和 discourse regimes，但不保证更优。不同上下文中的 Jacobian directions 可能互相抵消，导致通用平均稀释任务专用信号。

因此不建议一开始就只拟合一个混合 artifact。更好的顺序是分别拟合：

$$
J_{broad},\quad J_{task},\quad J_{onpolicy}
$$

然后在 broad、gold CoT 和 student rollout 上交叉测量 finite-difference fidelity。

确认性质以后，再考虑：

$$
J_{mix}
=
\alpha J_{broad}
+
\beta J_{task}
+
\gamma J_{onpolicy}
$$

混合时必须统一 sequence length、有效 positions 数和 target-horizon normalization，否则 mixture weight 会被序列长度暗中改变。

## 8. Training-Free Looped Transformers 的不变量

原文见 [Training-Free Looped Transformers，Section 2.3](https://arxiv.org/html/2605.23872#S2.SS3)。

### 8.1 Residual block 作为 Euler step

把一个 residual block 写成：

$$
g(x)=x+F_g(x)
$$

它可以被视为一个步长为 1 的 forward Euler step。

如果朴素地循环 $K$ 次：

$$
x_{k+1}=g(x_k)
$$

那么总积分时间近似从 1 变成了 $K$ 。下游层原本只训练过接收时间 1 的 endpoint，现在却收到近似时间 $K$ 的状态，因此容易崩塌。

### 8.2 Damped substeps

论文将每次更新缩小到原来的 $1/K$ ：

$$
x_{k+1}
=
x_k
+
\frac{1}{K}
(g(x_k)-x_k)
$$

等价地：

$$
x_{k+1}
=
\left(1-\frac{1}{K}\right)x_k
+
\frac{1}{K}g(x_k)
$$

于是总 step size 满足：

$$
\sum_{k=1}^{K}\Delta t_k=1
$$

循环只是更细地近似同一个 endpoint，而不是把计算继续推进到另一个深度时间。

论文还把原始 one-step output 作为 anchor。它的失败实验显示，仅仅固定 residual norm 不能挽救 naive looping，说明问题主要是方向和 endpoint 漂移，而不是 hidden norm。失败实验见 [Appendix G](https://arxiv.org/html/2605.23872#A7)。

### 8.3 与 lens 蒸馏的对应关系

Student 从第 $l$ 层到最终层存在一个真实 future map：

$$
h_L=F_{l\rightarrow L}(h_l)
$$

中间层改变一个小量后，最终 hidden 的变化近似为：

$$
\delta h_L
\approx
J_{s,l}\delta h_l
$$

当前 lens loss 直接要求中间 readout 靠近 teacher，却没有控制这个改变经过剩余 student layers 后会造成多大的 endpoint displacement。

因此，较强 lens supervision 可能产生：

$$
\text{提前到达 terminal-like state}
+
\text{剩余层继续执行完整 residual updates}
$$

这相当于 semantic-time overshoot。

小 aux weight 只形成局部扰动，所以模型不会立即离开原轨迹；但它的效果也容易小到落入随机种子波动。aux weight 一旦足以主导优化，就会破坏下游层期待的接口。

不能简单把论文中的 $1/K$ 复制成 lens weight。训练中的 raw loss coefficient 不是 forward ODE 的 step size。真正应该控制的是 aux update 造成的 final-policy displacement。

## 9. 三个可能被破坏的不变量

### 9.1 主不变量：depth-time / endpoint interface

相同 aux weight 并不意味着不同层、不同 arm 对最终输出施加了相同强度的扰动。

一个更合理的约束是：aux-only update 引起的最终策略变化不超过固定 budget。

例如：

$$
\mathbb E_x
\left[
D_{KL}
\left(
p_s^{before}(\cdot\mid x)
\;\|\;
p_s^{after\ aux\ step}(\cdot\mid x)
\right)
\right]
\le\epsilon
$$

这个量直接测量是否保持了原 student 的 endpoint interface。

### 9.2 Lens-score gauge

Lens 的 concept ranking 对下面的变换通常近似不敏感：

$$
z\mapsto az+b
$$

其中 $a$ 为正数。

但是 softmax KL 只对加上常数 $b$ 不变，对 scale $a$ 不变并不成立：

$$
\operatorname{softmax}(az)
\neq
\operatorname{softmax}(z)
$$

因此，两个模型可以具有相似的 top concepts，却具有不同的 entropy、max probability 和 vocabulary tail。Dense KL 会把这些差异全部当成必须修正的监督信号。

#### 单序列数值诊断

在一个真实 GSM8K 序列、32 个 completion positions 上测得：

| 层对，student / teacher | LL entropy，student / teacher | JL entropy，student / teacher | Full-vocab KL，B / C / E |
|---|---:|---:|---:|
| 6 / 8 | 10.60 / 11.00 | 3.37 / 8.02 | 2.72 / 5.51 / 6.38 |
| 12 / 16 | 10.16 / 11.03 | 2.79 / 3.99 | 2.85 / 7.84 / 4.54 |
| 18 / 24 | 7.15 / 8.14 | 1.44 / 3.01 | 3.44 / 4.21 / 4.19 |

虽然 student 和 teacher 的 J-lens logit standard deviation 大体都在 1.9 到 2.2，但 entropy 和 max probability 差异很大。因此这不是简单的 Jacobian matrix norm 爆炸。

C 的中层 dense KL 从 7.84 限制到 teacher top-200 后降到 3.76，说明大量梯度来自 vocabulary tail。但 E 的 top-200 改善很小，所以 tail 不是唯一原因。

同一诊断中：

- early E 的 hidden-gradient norm 大约是 B 的 2.6 倍；
- middle C 的 hidden-gradient norm 大约是 B 的 4.5 倍。

所以相同 aux weight 并不代表相同的 hidden forcing。

### 9.3 固定测量接口与线性化有效域

J-lens 是在 frozen student initialization 上拟合的：

$$
J_s^0
$$

训练过程中实际使用的对象却是：

$$
h_s(\theta),\quad N_s(\theta),\quad U_s(\theta)
$$

如果 $J_s^0$ 保持冻结，那么它对当前 student 的一阶近似会逐渐失效。

在 full fine-tuning 中，student final norm 和 LM head 也会更新。当前 2B student 还使用 tied input/output embeddings，因此 auxiliary loss 可以同时改变：

- 被测量的 hidden representation；
- 测量使用的 vocabulary codebook；
- 输入 embedding 所定义的部分表示坐标。

LoRA 下这些组件大体冻结，但 LoRA 实验也会受害，因此 moving head 是放大因素，不是唯一根因。

## 10. 现有训练证据

本地 Trackio 记录显示：

| Arm | aux weight | initial base loss | raw aux loss | weighted aux / base | pre-clip grad norm |
|---|---:|---:|---:|---:|---:|
| B | 0.1 | 0.037 | 3.10 | 8.3 | 2.39 |
| C | 0.1 | 0.037 | 5.89 | 15.8 | 3.00 |
| B | 0.01 | 0.037 | 3.10 | 0.83 | 0.75 |
| C | 0.01 | 0.037 | 5.89 | 1.58 | 0.77 |
| E | 0.01 | 0.037 | 4.82 | 1.29 | 0.85 |

所有 loss 和 gradient 都是有限的，没有出现 NaN 或 Inf。问题是 global gradient clipping 以后，较大的 auxiliary gradient 接管了更新方向。

后续 B weight sweep 从 0.003、0.03、0.1 到 0.3，准确率基本单调变差。最新 seed replication 也显示最好的 B、C、E 大多没有稳定超出随机种子波动，见 [`docs/experiments/results.md`](experiments/results.md)。

因此，初期 collapse 的直接触发因素确实是 loss scale，但更深层原因是 auxiliary objective 与最终任务需要的 student trajectory 存在方向冲突。

## 11. OPRD-Bridge 原文 recipe

原文见 [OPRD，Section 3.3](https://arxiv.org/abs/2606.06021) 和 [OPRD HTML](https://arxiv.org/html/2606.06021#S3.SS3)。

### 11.1 Stage 1：构造 bridge

首先按比例把每个 student layer 映射到 teacher layer：

$$
\phi(l_s)
=
\operatorname{round}
\left(
\frac{l_s-1}{L_s-1}
\cdot(L_t-1)
\right)+1
$$

这个映射显式对齐第一层和最后一层，中间层均匀分布。

在 2000 个 DAPO prompts 上用 student 生成 on-policy rollouts，同时收集 student 和 teacher hidden states。

对每个 teacher layer 的中心化 hidden 做 PCA：

$$
P_t^l
=
\operatorname{TopPCA}_r(h_t^l-\mu_t^l)
$$

形状为：

$$
P_t^l\in\mathbb{R}^{r\times d_t}
$$

然后学习一个 student projector：

$$
P_s^l\in\mathbb{R}^{r\times d_s}
$$

使得：

$$
P_s^lh_s^l
\approx
P_t^{\phi(l)}
\left(
h_t^{\phi(l)}-\mu_t^{\phi(l)}
\right)
$$

Stage 1 中两个 backbone 都冻结，只训练 student projectors。训练完成以后，teacher PCA projectors 和 student projectors 都冻结。

论文默认使用：

$$
r=8
$$

它报告 rank 8 的 projected cosine 大约为 95%，而增加 rank 后对齐质量下降。论文因此把低秩解释成共享结构本身的性质，而不只是计算压缩。

### 11.2 Stage 2：on-policy representation distillation

定义：

$$
q_s=P_sh_s
$$

$$
q_t=P_t(h_t-\mu_t)
$$

对两侧 projected vectors 做 L2 normalization：

$$
\widehat q_s=\frac{q_s}{\|q_s\|_2}
$$

$$
\widehat q_t=\frac{q_t}{\|q_t\|_2}
$$

再计算 MSE：

$$
L_{bridge}
=
\frac{1}{|\mathcal L|}
\sum_{l\in\mathcal L}
\frac{1}{|M|}
\sum_{i\in M}
\left\|
\widehat q_{s,l,i}
-
\widehat q_{t,\phi(l),i}
\right\|_2^2
$$

默认实验：

- 匹配所有 28 个 student layers；
- 使用最后 2000 个 response tokens；
- bridge 和 teacher 保持冻结；
- 只更新 student backbone；
- 使用 bridge representation objective，而不是把它当成一个极小正则项。

实验细节见 [OPRD Section 4.2.1](https://arxiv.org/html/2606.06021#S4.SS2.SSS1)。

## 12. OPRD 与当前 Arm D 的区别

| 项目 | OPRD 原文 | 当前 Arm D |
|---|---|---|
| 公共空间 | 双侧投影到 rank-8 空间 | teacher 到 student 的 full-rank affine map |
| 层 | 每个 student layer，包含 endpoints | 当前主要使用三个 layer pairs |
| bridge fit 数据 | 2000 个 student on-policy rollouts | in-repo fitter 使用较小 prompt 集；实际 artifact 来自外部 per-layer fit |
| normalization | projected vectors 做 L2 normalization | raw student-space hidden MSE |
| positions | 最后约 2000 个 response tokens | 通常最多 512 个 completion positions |
| 训练目标 | bridge representation objective | base OPD 加 bridge auxiliary loss |
| 当前权重 | 原文不是小正则项设计 | 当前 Arm D 使用约 1.94 |
| 接口 | 两侧 frozen low-rank projectors | frozen affine teacher-to-student bridge |

当前 Arm D 见 [`opdlens/arms.py`](../opdlens/arms.py)。in-repo bridge fitter 见 [`opdlens/fit/fit_bridge.py`](../opdlens/fit/fit_bridge.py)。

OPRD bridge 并不天然比 lens 更有 workspace 或 causal 意义。PCA top-variance directions 可能主要编码位置、格式、norm、routing 或其他共享统计。高 projected cosine 也不能证明这些方向就是 workspace。

但是 OPRD 至少显式处理了两个当前 B、C、E 缺少的问题：

- projector 在 Stage 2 中冻结，形成固定测量接口；
- projected vectors 先 normalization，使 loss 对整体 scale 不敏感。

低秩还限制了 student 通过 architecture-specific nuisance directions 过拟合 bridge。当前 full-rank raw-MSE Arm D 更像一个 dense OPRD-Bridge 变体，而不是原文 recipe。

## 13. 下一轮最有区分度的实验

### 13.1 第一优先级：gradient 和 endpoint audit

对同一 rollout 分别计算 base gradient 和 auxiliary gradient：

$$
g_{base}=\nabla_\theta L_{base}
$$

$$
g_{aux}=\nabla_\theta L_{aux}
$$

记录相对强度：

$$
\rho
=
\lambda
\frac{\|g_{aux}\|_2}
{\|g_{base}\|_2}
$$

记录方向冲突：

$$
c
=
\frac{
\langle g_{base},g_{aux}\rangle
}{
\|g_{base}\|_2\|g_{aux}\|_2
}
$$

需要按以下 parameter groups 分开记录：

- early、middle、late blocks；
- attention 和 MLP；
- embedding、final norm 和 LM head；
- 每一个 supervised layer 的独立 auxiliary term。

然后执行一个很小的 virtual aux-only step，测量最终 policy KL。这个实验可以直接检验 endpoint invariant，而不是只比较 scalar loss。

### 13.2 第二优先级：冻结 student readout

增加一个 opt-in knob，在 step 0 clone 并冻结 student 的：

$$
N_s^0,\quad U_s^0
$$

训练过程中始终用它们计算 lens auxiliary loss。

如果 full-FT collapse 明显减轻，说明 moving readout gauge 是重要原因。

### 13.3 第三优先级：gauge-respecting objective

一种简单控制是先对每个位置的 logits 做中心化和归一化：

$$
\widehat z
=
\frac{
z-\operatorname{mean}_V(z)
}{
\operatorname{std}_V(z)
}
$$

再匹配 cosine 或 MSE，而不是直接匹配 softmax probability。

这个目标对下面的正 scale 和 shift 变换不敏感：

$$
z\mapsto az+b
$$

另一个方向是只匹配 teacher-selected sparse concept margins 或 ranks，不匹配完整 vocabulary tail。

Top-k 只能作为诊断，不应被假定为通用修复。现有结果已经显示：它有时能补偿 B 的 harmful dense loss，但可能伤害 C 和 E 的 reverse-KL 配置。

### 13.4 第四优先级：测量 J-lens fidelity 是否随训练失效

在 step 0、50、100 等 checkpoint，对随机小扰动比较真实 future-map change：

$$
F_{l\rightarrow L}(h_l+\epsilon\delta h)
-
F_{l\rightarrow L}(h_l)
$$

与 offline Jacobian prediction：

$$
\epsilon J_l\delta h
$$

记录 cosine similarity 和 relative error。

如果大 auxiliary run 的 J-lens fidelity 在 eval accuracy 坍塌之前先下降，就支持 probe hacking / linearization-domain 假设。

### 13.5 第五优先级：校准语料三分实验

分别拟合：

$$
J_{broad},\quad J_{gold},\quad J_{onpolicy}
$$

统一：

- prompt 数；
- sequence length；
- 有效 source 和 target positions；
- future-target normalization；
- fitted layers。

先构造 cross-fidelity matrix，再决定训练时使用哪个 artifact。不要只根据 unigram coverage 或某个单一 downstream accuracy 选择。

### 13.6 最后再尝试 ODE-inspired loss

一个方向是匹配 lens trajectory 的速度，而不是强制绝对中间状态相同：

$$
v_k
=
\frac{
r_{l_{k+1}}-r_{l_k}
}{
\tau_{k+1}-\tau_k
}
$$

其中 $r_l$ 是某个经过 gauge normalization 的 workspace readout， $\tau_l$ 是归一化深度时间。

另一种更直接的方法是按 aux-only step 实际造成的 final-policy KL 自适应缩放 auxiliary update，使其保持固定 endpoint budget。

这两种方法都比简单把 raw auxiliary weight 除以层数更接近 looped-transformer 论文真正保持的量。

## 14. 可证伪预测

如果本文的主判断成立，那么在 accuracy 坍塌之前应当先观察到：

1. base gradient 与 auxiliary gradient 的 cosine 下降，甚至变成负数；
2. 一个很小的 aux-only parameter step 引起的 final-policy KL 急剧增加；
3. offline student Jacobian 对真实 finite perturbation 的预测 fidelity 下降；
4. lens loss 继续下降，但 base loss、最终 teacher agreement 或 task accuracy 开始恶化；
5. 冻结 readout、限制 endpoint displacement 或使用 gauge-invariant objective 能显著提高可用 auxiliary strength。

如果这些现象都没有出现，那么 depth-time / endpoint-interface 假设就不成立，应转而检查数据采样、position alignment、loss mask、KL direction 或 optimizer interaction。
