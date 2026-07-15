# 基于 teacher J-lens 的稀疏 concept supervision

## 1. 动机

当前 lens auxiliary loss 会在很大的词表上匹配 teacher 和 student 的分布。
即使使用 top-k 截断，例如令 $k = 100$，它仍然要求 student 复现一个相当
dense 的 probe distribution。大量优化预算可能因此花在弱信号、同义重复或
probe-specific 的 token directions 上。

一个更保守的替代方案是：在每个候选位置，让 teacher 只产生零个、一个或两个
concept labels。

- 如果 teacher J-lens readout 足够尖锐，则启用一个稀疏的辅助分类任务；
- 否则将该位置的 auxiliary loss 严格设为零，回退到 base OPD 路径。

这样会改变 J-lens 的角色。它不再提供一个需要被完整拟合的 dense target
distribution，而只充当 teacher-only concept selector。

仅有尖锐度还不足以证明一个 concept 值得监督。某个 token 可能很容易被 probe
读出，但真实下游计算对它并不敏感。下面的三个方案因此逐步采用更强的 concept
有效性标准：

1. teacher distribution 的置信度；
2. concept 对 teacher endpoint 的因果作用；
3. concept 是否能通过 student 的真实 output path 被学习和报告。

## 2. 共享的 concept gate

令 $z \in \mathbb{R}^{|\mathcal V|}$ 表示某个候选 layer 和 position 上、已经
detach 的 teacher J-lens logits。使用 selection temperature
$\tau_{\mathrm{sel}}$ 定义：

$$
p(v)
=
\frac{\exp(z_v / \tau_{\mathrm{sel}})}
{\sum_{u \in \mathcal V} \exp(z_u / \tau_{\mathrm{sel}})}.
$$

在排序前，应过滤 control tokens、纯空白 tokens 和纯标点 tokens，但保留数字、
运算符、单位和数学术语。还应合并明显的 tokenizer variants，例如前导空格和
大小写变体。对于数值 concept，只归一化表面格式，不能把不同数值合并为同一个
concept。

### 2.1 尖锐度统计量

分布的有效支持集大小定义为：

$$
N_{\mathrm{eff}}
=
\exp\bigl(H(p)\bigr),
$$

其中：

$$
H(p)
=
-\sum_{v \in \mathcal V} p(v) \log p(v).
$$

J-lens 的 logit scale 可能随模型和 layer 改变，因此不能只根据 raw probability
决定是否接受。令 $z_1 \geq z_2 \geq z_3$ 表示三个最大的有效 concept logits，
定义经过 robust scale 标准化的 margins：

$$
m_1
=
\frac{z_1 - z_2}
{\operatorname{MAD}(z_{\mathrm{top}\text{-}64}) + \epsilon}
$$

以及：

$$
m_2
=
\frac{z_2 - z_3}
{\operatorname{MAD}(z_{\mathrm{top}\text{-}64}) + \epsilon}.
$$

这里的 $\operatorname{MAD}$ 是 top-64 有效 logits 的 median absolute
deviation。

### 2.2 选择零个、一个或两个 concepts

selector 返回一个集合 $S$：

$$
S
=
\begin{cases}
\{y_1\},
& p_1 \text{ 足够高且 } m_1 \text{ 足够大}, \\
\{y_1, y_2\},
& p_1 + p_2 \text{ 足够高，} m_2 \text{ 足够大，且 } p_2 / p_1
  \text{ 不太小}, \\
\varnothing,
& \text{其他情况}.
\end{cases}
$$

选择 top-2 应当表示两个 concepts 作为一个整体与词表其余部分明显分离。不能仅仅
因为 top-1 没通过 margin 判据，就把一个很弱的 runner-up 提升为第二个 concept。

第一轮实验应只允许 top-1。只有 top-1 supervision 已经表现出价值后，才应通过
独立 knob 引入 top-2。

### 2.3 阈值校准

类似 $p_1 > 0.5$ 的绝对阈值很可能无法跨 layer 复用。更稳妥的办法是在 train-only
rollouts 上校准阈值，使 gate 达到固定的 acceptance coverage，例如 5%、10% 和
20%。这样不同方法可以在相近的 auxiliary opportunity 下比较，同时完全不改 eval。

可选的稳定性判据包括：

- 同一个归一化 concept 在相邻三个 positions 中至少出现两次；
- 在两个由不同 calibration shards 拟合的 J-lenses 上得到相同 concept；
- 在同一个 workspace band 的相邻 layers 上得到相同 concept。

这些判据都应当是 knobs，因为有意义的中间 concept 也可能只短暂出现。

### 2.4 严格的 baseline fallback

令 $g_i \in \{0, 1\}$ 表示候选位置 $i$ 的 gate，$w_i$ 表示一个可选的、已经
detach 的 confidence weight。batch auxiliary loss 应定义为：

$$
L_{\mathrm{aux}}
=
\frac{1}{N_{\mathrm{candidate}}}
\sum_{i=1}^{N_{\mathrm{candidate}}}
g_i w_i \ell_i.
$$

不要除以 accepted positions 的数量。如果按 accepted count 归一化，那么即使只有
一个极少见的位置通过 gate，它仍会得到完整的 auxiliary weight；gate 变严格也不会
减少总辅助压力。

如果所有 $g_i = 0$，那么应有 $L_{\mathrm{aux}} = 0$，该 update 就是完全不变的
base OPD update。

## 3. 方案 A：sharpness-gated sparse classification

这是成本最低的方案，用来检验 dense full-vocabulary matching 是否是当前失败的
主要原因。

对每个被接受的 teacher concept $y$，只计算对应的 student score 和一小组 negative
scores。令 $s_y$ 表示 positive concept 的 student score，令 $\mathcal N_i$ 包含
16 到 64 个 negatives。可以使用一个会饱和的 ranking loss：

$$
\ell_i
=
\sum_{y \in S_i}
\widetilde w_y
\operatorname{softplus}
\left(
\gamma - s_y
+ \operatorname{logsumexp}_{n \in \mathcal N_i} s_n
\right),
$$

其中 $\gamma$ 是 margin，$\widetilde w_y$ 是 teacher probability 在一到两个
selected concepts 上重新归一化后的权重。

与 full cross-entropy 或 KL 不同，当 student 达到目标 margin 后，这个 loss 会接近
零，不会持续把一个已经正确的 concept 推得越来越尖。

negative candidates 可以包括：

- teacher 中紧随 selected concepts 之后的若干 ranks；
- 随机采样的有效 concept tokens；
- batch 内其他样本的 positive concepts。

positive concept 的 tokenizer variants 和明显 lexical aliases 不应被当作 negatives。
这些 negative tokens 只是分类边界的竞争者，不是额外 teacher targets。

### 3.1 算力开销

这个方案不需要额外的 transformer forward。

teacher 仍需通过一次 vocabulary readout 找到 top concepts，但可以先将每条 sequence
的候选位置限制到 8 或 16 个，而不是最多 512 个。student 只需计算一到两个
positives 和几十个 negatives，从而避免构造带梯度的
$[N, |\mathcal V|]$ tensor。

### 3.2 该方案能检验什么

方案 A 直接检验当前失败是否来自对 dense、低置信度 vocabulary tail 的匹配。它
不会消除固定的 student observer，因此 student J-lens drift、gauge dependence 和
probe hacking 仍然可能存在。

## 4. 方案 B：causally screened sparse classification

这个方案只有在移除某个 concept 会改变 teacher 的真实 endpoint behavior 时，才
接受该 concept。

首先使用方案 A 的低成本 sharpness gate 做预筛选。对于 top-1 concept，其 teacher
layer direction 为 $v_y$，可以移除当前 activation 在该方向上的坐标：

$$
h'
=
h - v_y \bigl(v_y^\dagger h\bigr).
$$

对于两个 concepts，令：

$$
V
=
\begin{bmatrix}
v_{y_1} & v_{y_2}
\end{bmatrix},
$$

然后移除它们的联合分量：

$$
h'
=
h - V V^\dagger h.
$$

使用 intervention 再运行一次 frozen teacher，并在一组未来 completion positions
$\mathcal F$ 上比较 clean 和 ablated final distributions：

$$
C_y
=
\frac{1}{|\mathcal F|}
\sum_{j \in \mathcal F}
D_{\mathrm{KL}}
\left(
p_{\mathrm{clean}, j}
\,\|\,
p_{\mathrm{ablated}, j}
\right).
$$

仅有绝对 endpoint change 还不够，因为某些 layers 可能对任何 matched-norm
perturbation 都很敏感。应当同时使用具有相同 perturbation norm 的 random 或
non-J-space control directions：

$$
R_y
=
\frac{C_y}
{\operatorname{median}(C_{\mathrm{control}}) + \epsilon}.
$$

只有同时满足以下条件时才接受 concept：

- distributional sharpness gate 通过；
- $C_y$ 超过在 train data 上校准的绝对阈值；
- $R_y$ 超过相对 causal threshold。

接受后的 label 可以使用与方案 A 相同的 sparse student ranking loss。

### 4.1 算力开销

teacher intervention 不需要 backward，只需要对通过 cheap prefilter 的样本增加一次
no-grad teacher forward。

可以通过以下方式控制成本：

- 将 prefilter coverage 限制为 5% 或 10%；
- 每条 sequence 最多验证一个 layer 和一个 concept set；
- 沿 batch dimension 合并多个 ablation 和 control variants；
- 在现有 teacher-forced sequence 上比较 logits，不生成新 continuation；
- 先在固定的 train replay buffer 上运行 causal screen，再用结果校准更便宜的
  online gate。

如果 10% 的 sequences 各增加一次 teacher forward，预期 teacher-forward overhead
约为 10%。加入一到两个 matched controls 后，取决于 batching efficiency，额外成本
可能约为 20% 到 30%。

### 4.2 该方案能检验什么

方案 B 可以区分仅仅 probe-visible 的 teacher token 和在当前 context 中确实
load-bearing 的 token。它提供的 labels 比方案 A 更可靠，但 student 仍可能通过
stale 或 hackable probe 满足 sparse classification objective。

## 5. 方案 C：counterfactual concept-report classification

这个方案把 teacher-selected concept 用作 training-only report target，而不是对
student intermediate hidden 直接施加 probe loss。

在通过 gate 的 prefix 后追加一个很短、只在训练时存在的问题，例如：

> At this point, name the single intermediate concept most relevant to
> completing the task.

target Assistant response 只包含 teacher 选出的一个或两个 concept tokens。原始、
不中断的 rollout 继续使用正常 base OPD loss。auxiliary branch 只在短 concept
response 上使用普通 final-output supervision：

$$
L
=
L_{\mathrm{base}}(x)
+ \lambda g(x)
L_{\mathrm{report}}
\bigl(x \mathbin{\|} q_{\mathrm{report}} \rightarrow S(x)\bigr).
$$

这里 $x$ 是原始 prefix，$q_{\mathrm{report}}$ 是追加的 training-only question，
$S(x)$ 是 teacher-selected concept set。

eval 和正常 generation 中都不出现 report question。gate 拒绝 concept 时，不创建
report branch，该样本严格沿 baseline path 执行。

### 5.1 算力开销

J-lens selector 已经给出了 target labels，因此不需要额外 teacher generation。每个
accepted example 需要额外一次 student forward 和 backward，输入为 prefix 加上一个
很短的 report turn。

当梯度需要流经 prefix 时，训练阶段很难直接复用 KV cache。可以通过以下方式控制
成本：

- 每条 sequence 最多创建一个 report branch；
- 从 5% 或 10% gate coverage 开始；
- 只使用一个随机或最高置信度 truncation point；
- 保持 question 和 response 很短。

在 10% coverage 下，约有额外 10% 的 student branches 需要处理，但实际 wall-clock
overhead 还取决于 prefix length 和 batching。

### 5.2 该方案能检验什么

方案 C 不依赖 fixed student Jacobian、student layer mapping 或 dense intermediate
vocabulary loss。梯度会经过 student 当前真实的 downstream layers 和正常 output
head。因此，它更直接地与 endpoint function 对齐，并且不会惩罚那些保持模型输出
函数不变的 hidden-coordinate changes。

主要风险是 student 只学会回答人工 report question，却没有把 concept 迁移到
uninterrupted reasoning。这仍然是一个干净、可证伪的 counterfactual reflection
training 变体。

## 6. 三种方案对比

| 方案 | Concept 判据 | Student objective | 额外 transformer 算力 | 主要残余风险 |
|---|---|---|---|---|
| A. Sharpness-gated | Teacher confidence 和可选的局部稳定性 | Sparse intermediate ranking | 无额外 forward | Student 可以通过 stale probe 满足 loss |
| B. Causally screened | Confidence 加 teacher endpoint intervention | Sparse intermediate ranking | 对少量样本增加 no-grad teacher forwards | Teacher causality 不保证 student-probe fidelity |
| C. Counterfactual report | Teacher confidence 和可选的局部稳定性 | 在 training-only branch 上做真实 final-output classification | 对 accepted examples 增加 student forward/backward | Report skill 可能不迁移到 uninterrupted behavior |

## 7. 推荐实验顺序

第一步运行方案 A，只允许 top-1，并比较 5% 和 10% coverage。这是检验 dense
vocabulary tail 是否导致当前失败的最低成本实验。

同时在 matched coverage 下运行方案 C，作为使用真实 endpoint path 的对照。如果
方案 C 有效而方案 A 无效，那么更可能是 intermediate fixed-observer objective 有
问题，而不是 concept labels 本身无效。

方案 B 可以先作为 train-only causal audit，在数百到一千个 contexts 上运行。检查
confidence、normalized margin、effective support 和 persistence 是否能预测 endpoint
effect。如果可以，再用结果校准便宜的 gate；如果完全不能，则不应扩大
sharpness-only supervision。

所有实验都应保留现有 eval path，并记录：

- 每个 layer 的 gate coverage；
- top-1 和 top-2 acceptance rates；
- concept frequency 和 duplicate-token rate；
- accepted 与 rejected 样本的 sharpness statistics；
- auxiliary gradient norm 及其对 clipping budget 的占用；
- audited subset 上的 causal endpoint effect；
- base loss、auxiliary loss 和最终 task metrics。

核心对比不只是 sparse supervision 与 dense supervision。更重要的是，teacher concept
究竟只是由 probe confidence 选出、经过了真实 causal use 验证，还是通过 student
自己的 endpoint path 被学习。

## 8. 参考

- Anthropic, [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
- Anthropic, [Technical details of J-lens use cases](https://transformer-circuits.pub/2026/workspace/index.html#methods-technical-details)
- Anthropic, [Counterfactual Reflection Training](https://transformer-circuits.pub/2026/workspace/index.html#reflection)
