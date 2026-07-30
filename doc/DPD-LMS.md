# DPD-LMS逐样点补偿与系数更新原理

本文解释 `inc/lib/DpdLms.py` 中基于GMP特征的复数LMS/NLMS数字预失真算法。重点不是再次介绍PA静态逆，而是回答以下工程问题：

1. 每收到一个样点时怎样构造特征、计算误差并更新一次系数；
2. 逐样点算法与 `DpdGmp` 批量岭回归在程序结构上有什么本质区别；
3. 为什么同步仍按整帧完成，而系数可以按样点完成；
4. 为什么推荐“影子系数逐样点更新、当前发送系数逐帧提交”；
5. 哪些数值保护是高阶GMP在线更新必须增加的。

类的完整参数和调用方式见 [DpdLms.md](./DpdLms.md)。批量GMP原理见 [DPD-GMP.md](./DPD-GMP.md)。

---

## 1. 逐样点DPD-LMS解决什么问题

批量岭回归收集一整批参考和标签后求解一次：

```math
\mathbf c
=
\left(
\boldsymbol{\Phi}^{H}\boldsymbol{\Phi}
+
\lambda\mathbf I
\right)^{-1}
\boldsymbol{\Phi}^{H}\mathbf d.
```

它在固定数据集上精度高，但必须积累正规矩阵并在批末求解线性方程。

LMS不构造完整正规矩阵。第 $n$ 个样点到达后立即执行：

```math
\mathbf c[n]
\longrightarrow
\mathbf c[n+1].
```

它的主要价值不是在静态数据上取代批量最小二乘，而是：

- 避免 $K\times K$ 复矩阵求解；
- 允许PA温度、增益、偏置和负载缓慢变化时持续跟踪；
- 更新结构适合DSP或FPGA流水线；
- 可以明确观察每个样点对系数的贡献。

静态、无噪声、同一训练集条件下，批量岭回归通常仍是精度上限。逐样点算法的优势是更新连续、存储小和跟踪能力强。

---

## 2. 与批量DpdGmp完全相同的GMP模型

DPD输出写为：

```math
\hat d[n]
=
\sum_{k=0}^{K-1}
c_k[n]\phi_k[n]
=
\boldsymbol{\phi}^{T}[n]\mathbf c[n].
```

`DpdLms`直接继承 `DpdGmp`，所以主项、滞后包络项和超前包络项的定义与系数顺序完全一致。

主项：

```math
\phi_{p,m}^{\mathrm{main}}[n]
=
x[n-m]|x[n-m]|^{p-1}.
```

滞后包络项：

```math
\phi_{p,m,l}^{\mathrm{lag}}[n]
=
x[n-m]|x[n-m-l]|^{p-1}.
```

超前包络项：

```math
\phi_{p,m,l}^{\mathrm{lead}}[n]
=
x[n-m-l]|x[n-m]|^{p-1}.
```

“超前”只表示载波样点和包络样点的相对索引，所有样点仍来自当前或过去，代码没有读取未来数据。

默认阶数、主记忆和交叉记忆为：

```math
\mathcal P=\{1,3,5,7\},
\qquad
M=3,
\qquad
L=2.
```

主项数量为：

```math
K_{\mathrm{main}}=4\times3=12.
```

交叉项只对3、5、7阶生成：

```math
K_{\mathrm{cross}}
=
2\times3\times3\times2
=
36.
```

所以默认每个样点需要更新：

```math
K=48
```

个复系数。增广共轭GMP若以后接入相同更新器，特征数将加倍。

---

## 3. 复数LMS的逐样点推导

第 $n$ 个样点的目标为 $d[n]$，更新前预测为：

```math
\hat d[n]
=
\boldsymbol{\phi}^{T}[n]\mathbf c[n].
```

误差定义为：

```math
e[n]
=
d[n]-\hat d[n].
```

瞬时平方误差为：

```math
J[n]
=
|e[n]|^2.
```

当前程序采用 $\boldsymbol{\phi}^{T}\mathbf c$ 的系数约定，而不是 $\mathbf c^{H}\boldsymbol{\phi}$。对复系数使用Wirtinger梯度可得到下降方向：

```math
-\frac{\partial J[n]}{\partial\mathbf c^{*}}
=
\boldsymbol{\phi}^{*}[n]e[n].
```

所以复数LMS更新为：

```math
\mathbf c[n+1]
=
\mathbf c[n]
+
\mu
\boldsymbol{\phi}^{*}[n]e[n].
```

这里：

- 星号表示逐元素复共轭；
- $\mu$ 是 `learningRate`；
- 预测必须使用更新前系数；
- 同一样点的误差计算完成后才能覆盖系数。

对于理想宽平稳输入，均值收敛的一般边界为：

```math
0<\mu<
\frac{2}
{\lambda_{\max}(\mathbf R_{\phi})},
```

其中：

```math
\mathbf R_{\phi}
=
E[
\boldsymbol{\phi}[n]
\boldsymbol{\phi}^{H}[n]
].
```

高阶GMP特征相关性强，而且各列能量差异大，因此普通LMS的稳定步长很难从一个固定数值直接确定。

---

## 4. 为什么默认使用NLMS

NLMS用当前特征总能量归一化更新：

```math
\mathbf c[n+1]
=
\mathbf c[n]
+
\frac{
\mu
\boldsymbol{\phi}^{*}[n]e[n]
}{
\delta+
\|\boldsymbol{\phi}[n]\|_2^2
}.
```

其中 $\delta$ 对应 `normalizationEpsilon`，用于避免静默区和包络过零时分母接近零。

理想NLMS的常用步长范围为：

```math
0<\mu<2.
```

工程默认值为：

```math
\mu=0.05.
```

该值比理论上限保守，因为真实反馈还包含：

- 同步估计残差；
- ADC量化和反馈噪声；
- 高阶特征相关；
- PA工作点变化；
- 输出包络限幅。

如果收敛很慢，可逐步提高到0.1或0.2；如果NMSE振荡、系数范数持续增长或ACLR恶化，应减小步长。

---

## 5. 高阶特征必须进行逐列尺度归一化

仅使用特征总能量仍不足以解决不同阶次的量纲差异。当归一化包络小于1时：

```math
|x|^7\ll|x|.
```

第 $k$ 个特征的RMS尺度定义为：

```math
s_k
=
\sqrt{
\frac{1}{N}
\sum_{n=0}^{N-1}
|\phi_k[n]|^2
}.
```

定义归一化特征和归一化系数：

```math
z_k[n]
=
\frac{\phi_k[n]}{s_k},
```

```math
\tilde c_k[n]
=
s_kc_k[n].
```

于是预测保持不变：

```math
\boldsymbol{\phi}^{T}[n]\mathbf c[n]
=
\mathbf z^{T}[n]\tilde{\mathbf c}[n].
```

程序实际更新 $\tilde{\mathbf c}$：

```math
\tilde{\mathbf c}[n+1]
=
\tilde{\mathbf c}[n]
+
\frac{
\mu
\mathbf z^{*}[n]e[n]
}{
\delta+\|\mathbf z[n]\|_2^2
}.
```

更新后再转换回部署系数：

```math
c_k[n+1]
=
\frac{\tilde c_k[n+1]}{s_k}.
```

### 5.1 帧尺度模式

`featureScaleMode="frame"`先对完整参考帧做一次只读能量遍历，得到 $s_k$，然后在整个逐样点更新过程中冻结尺度。

优点：

- 每个系数的物理含义在一帧内不变化；
- 与批量DpdGmp的列归一化定义一致；
- 收敛曲线更容易复现；
- 适合仪表返回完整帧后在软件中逐点回放。

这一步只统计特征能量，不计算正规矩阵，也不求解系数。

### 5.2 运行尺度模式

`featureScaleMode="running"`不需要提前获得完整帧。每个样点更新指数功率：

```math
P_k[n]
=
\beta P_k[n-1]
+
(1-\beta)|\phi_k[n]|^2.
```

尺度为：

```math
s_k[n]
=
\sqrt{\max(P_k[n],\delta)}.
```

$\beta$ 对应 `featurePowerForgettingFactor`，默认0.999。尺度变化时，程序每个样点都先执行：

```math
\tilde c_k[n]=s_k[n]c_k[n],
```

再更新并除回新尺度。这个特殊转换保证“改变坐标尺度”不会被错误解释成“改变DPD物理系数”。

---

## 6. 一个样点在程序内部经历什么

`UpdateSampleFloating`严格按以下顺序执行：

1. 把当前参考样点压入最新样点位于索引0的历史缓存；
2. 根据 `featureSpecs` 构造一行GMP特征；
3. 用更新前影子系数计算预测；
4. 计算目标减预测的复误差；
5. 累积在线误差功率和目标功率；
6. 检查 `updateDecimation` 和样点权重；
7. 得到帧冻结或运行特征尺度；
8. 转换到归一化系数坐标；
9. 计算LMS或NLMS梯度；
10. 施加恒等先验泄漏；
11. 限制单样点系数变化范数；
12. 更新影子系数；
13. 若选择样点提交模式，把完整影子向量复制到活动向量。

时序可以写成：

```text
new x[n]
    |
    v
shift causal history
    |
    v
build one phi[n]
    |
    v
predict with c[n]
    |
    v
e[n] = d[n] - prediction
    |
    v
normalize and update shadow c[n+1]
    |
    +--> sample commit: active = shadow immediately
    |
    +--> frame commit: active remains unchanged
```

历史缓存长度不是完整帧长度，而是：

```math
N_{\mathrm{history}}
=
1+
\max_k(m_k+l_k).
```

因此样点内存与帧长度无关。

---

## 7. 为稳定性增加的特殊处理

### 7.1 静默样点跳过更新

当前特征能量为零时，预测和目标仍进入在线NMSE统计，但系数不更新。否则全零特征只会消耗运算，并可能让数值保护项主导更新。

### 7.2 样点权重上限

峰值加权或外部权重先按帧归一化，再限制为：

```math
w[n]\leq w_{\max}.
```

`maximumSampleWeight`默认8。它防止单个异常峰值把NLMS稳定步长瞬间放大很多倍。

### 7.3 泄漏指向恒等DPD

普通泄漏把系数拉向零，但DPD的安全初值应是恒等映射。因此程序使用：

```math
\Delta\tilde{\mathbf c}_{\mathrm{leak}}[n]
=
-\mu\lambda
\left(
\tilde{\mathbf c}[n]
-
\tilde{\mathbf c}_{0}
\right),
```

其中 $\tilde{\mathbf c}_{0}$ 只有零时延一阶主项为1。`leakageFactor`默认很小，只抑制长期随机漂移。

### 7.4 单样点更新范数投影

若原始系数步进为 $\Delta\mathbf c[n]$，并且：

```math
\|\Delta\mathbf c[n]\|_2
>
\Delta c_{\max},
```

则执行：

```math
\Delta\mathbf c_{\mathrm{safe}}[n]
=
\Delta\mathbf c[n]
\frac{
\Delta c_{\max}
}{
\|\Delta\mathbf c[n]\|_2
}.
```

该限制对应 `maximumSampleUpdateNorm`。它主要保护同步失锁、反馈突发毛刺和错误定点码值，不应被用来掩盖长期步长过大。

### 7.5 训练梯度不穿过输出硬限幅

`maximumOutputMagnitude`仍用于部署 `Process`，但LMS瞬时预测和梯度使用限幅前线性参数模型。硬限幅不可微，如果直接用限幅后的预测却仍使用线性GMP梯度，会造成错误梯度。训练完成后的固定NMSE评估重新应用部署限幅，以反映实际输出。

---

## 8. 影子系数和活动系数

程序维护两套向量：

```math
\mathbf c_{\mathrm{adaptive}}
```

和：

```math
\mathbf c_{\mathrm{active}}.
```

### 8.1 帧提交模式

`coefficientCommitMode="frame"`时，每个样点只修改影子系数：

```math
\mathbf c_{\mathrm{adaptive}}[n+1]
\ne
\mathbf c_{\mathrm{adaptive}}[n],
```

而当前发送帧始终使用：

```math
\mathbf c_{\mathrm{active}}
=
\mathrm{constant}.
```

帧末执行：

```math
\mathbf c_{\mathrm{active}}
\leftarrow
\mathbf c_{\mathrm{adaptive}}.
```

这是默认模式。它避免OFDM符号内部系数变化把DPD变成快速时变系统，从而产生额外带内调制和带外频谱。

### 8.2 样点提交模式

`coefficientCommitMode="sample"`在每次更新后执行：

```math
\mathbf c_{\mathrm{active}}[n+1]
\leftarrow
\mathbf c_{\mathrm{adaptive}}[n+1].
```

这种模式适合：

- 验证严格在线算法；
- 对照硬件流水线；
- 研究时变系数的影响。

它不应默认用于Wi-Fi发射，因为同一帧内的系数调制可能使ACLR变差。

---

## 9. 间接学习中的同步为什么仍按整帧完成

间接学习使用实际PA输入 $u[n]$ 和反馈PA输出 $y[n]$。反馈先经过 `SigProc`：

```math
y[n]
\longrightarrow
\tilde y[n],
```

其中补偿：

- 整数时延；
- 分数时延；
- 载波频偏；
- 采样频偏；
- 公共复增益。

后置逆预测为：

```math
\hat u[n]
=
\boldsymbol{\phi}^{T}(\tilde y[n])
\mathbf c[n].
```

误差为：

```math
e[n]
=
u[n]-\hat u[n].
```

再按样点更新系数。

同步不能只用一个样点估计。相关峰、CFO相位斜率、SFO时间漂移和公共复增益都需要一段数据。因此 `UpdateIndirect` 的流程是：

```text
arbitrary-length feedback capture
    |
    v
frame-level SigProc estimation and alignment
    |
    v
one aligned output sample for each PA-input sample
    |
    v
chronological sample-by-sample LMS updates
```

反馈记录可以比PA输入更长，前后补零和仪表触发延迟由 `SigProc` 处理，不要求两个原始数组长度相等。

真正的无限流在线系统需要把已估计的时延、CFO和SFO变成持续跟踪器，并用FIFO保存与反馈时延对应的PA输入样点。这是实时硬件接收链的职责，不应在单个LMS乘加核中重复实现。

---

## 10. 为什么不能直接用PA输出误差做普通LMS

若直接定义：

```math
e_y[n]=x[n]-y[n],
```

PA输出满足：

```math
y[n]
=
F_{\mathrm{PA}}
\left\{
\boldsymbol{\phi}^{T}[n]\mathbf c[n]
\right\}.
```

输出对系数的导数还包含PA局部雅可比：

```math
\frac{\partial y[n]}{\partial\mathbf c}
=
J_{\mathrm{PA}}[n]
\boldsymbol{\phi}[n].
```

忽略 $J_{\mathrm{PA}}$ 的更新方向不是真正输出误差梯度。直接学习应使用Filtered-X思想：

```math
\boldsymbol{\psi}[n]
=
J_{\mathrm{PA}}[n]
\boldsymbol{\phi}[n],
```

```math
\mathbf c[n+1]
=
\mathbf c[n]
+
\frac{
\mu
\boldsymbol{\psi}^{*}[n]e_y[n]
}{
\delta+\|\boldsymbol{\psi}[n]\|_2^2
}.
```

当前第一版采用间接学习或外部标签学习，不假装普通LMS已经包含PA雅可比。频响测量和局部雅可比稳定后，可以在后续版本增加Filtered-X直接学习。

---

## 11. 逐样点与批量处理的程序实现差异

| 实现位置 | 批量 `DpdGmp` | 逐样点 `DpdLms` | 特殊处理原因 |
|---|---|---|---|
| 数据可用性 | 需要完整参考和标签 | `UpdateSample`只需要当前样点和历史 | 支持在线流 |
| 基函数 | `BuildGmpBasisChunk`构造多行矩阵 | `BuildFeatureVector`只构造一行 | 避免每个样点重复创建矩阵 |
| 历史 | 由数组切片和零填充得到 | `sampleHistory`环形逻辑缓存 | 保留因果记忆 |
| 列尺度 | 第一遍完整数据统计 | 帧预统计或运行指数功率 | 高阶特征尺度稳定 |
| 求解 | 累积 $K\times K$ 正规矩阵后求解 | 每个样点执行向量更新 | 不需要矩阵逆 |
| 系数状态 | 批末产生一个新向量 | 每个样点产生一个影子向量 | 支持漂移跟踪 |
| 部署 | 求解后整体替换 | 帧提交或样点提交 | 避免帧内时变频谱 |
| 正则 | 岭矩阵和先验中心 | 恒等泄漏和单步投影 | 在线算法没有批量正规矩阵 |
| 同步 | 可在训练前一次完成 | 仍在样点循环前按帧完成 | 单样点无法估计同步量 |
| 指标 | 更新前/后固定模型NMSE | 更新前、在线、更新后三种NMSE | 在线误差使用不断变化的系数 |
| 计算量 | 每批约 $O(NK^2+K^3)$ | 每样点约 $O(K)$ | 适合流水线但需要更多样点收敛 |
| 存储量 | 正规矩阵约 $O(K^2)$ | 系数、尺度和历史约 $O(K)$ | 特征历史与帧长无关 |

### 11.1 三种NMSE不能混用

`beforeNmseDb`使用训练前固定系数扫过整帧。

`onlineNmseDb`按样点累积：

```math
\mathrm{NMSE}_{\mathrm{online}}
=
10\log_{10}
\frac{
\sum_n w[n]|d[n]-\hat d[n;\mathbf c[n]]|^2
}{
\sum_n w[n]|d[n]|^2
}.
```

这里每个样点使用不同系数。

`afterNmseDb`使用最终固定影子系数重新扫过整帧：

```math
\mathrm{NMSE}_{\mathrm{after}}
=
10\log_{10}
\frac{
\sum_n w[n]|d[n]-\hat d[n;\mathbf c_{\mathrm{final}}]|^2
}{
\sum_n w[n]|d[n]|^2
}.
```

比较部署模型时应使用 `afterNmseDb`；观察训练过程平均误差时使用 `onlineNmseDb`。

---

## 12. 定点边界

与其他模块一致：

- `width=0`时公开样点是归一化浮点复数；
- `width>0`时 `UpdateSample` 的参考、目标和返回预测都是I/Q整数码；
- 系数、特征幂、归一化和误差更新始终在内部浮点完成；
- `Process`仍返回相同公开数据约定。

逐样点硬件移植时可进一步对系数和梯度定点化，但这超出当前“公开接口定点、内部算法浮点”的工程边界。应单独确定：

- 特征乘法位宽；
- 系数小数位；
- 梯度累加保护位；
- NLMS倒数近似；
- 饱和与舍入位置。

---

## 13. 计算量和实时性

默认48个特征时，每个有效更新样点主要需要：

- 48项GMP特征计算；
- 48项复乘加完成预测；
- 48项归一化和梯度更新；
- 一次特征能量求和与标量倒数。

Python实现用于算法验证、离线回放和硬件黄金参考，不保证在80、320或640 MS/s下实时逐样点运行。实际Wi-Fi实时实现需要：

- C/C++或Numba加速的软件路径；或者
- DSP向量核；或者
- FPGA并行/时分复用流水线。

`updateDecimation>1`可以降低自适应更新速率，但 `Process` 的DPD推理仍需处理每个发送样点。默认 `updateDecimation=1`用于严格逐样点参考。

---

## 14. 收敛判断和调参顺序

推荐按以下顺序：

1. 先用 `(1,3)`、`memoryDepth=1`、`crossMemoryDepth=0` 验证系数方向；
2. 使用 `featureScaleMode="frame"` 排除运行尺度变化；
3. 关闭泄漏，使用无噪声已知标签验证；
4. 从 `learningRate=0.05` 开始；
5. 检查 `afterNmseDb` 是否稳定下降；
6. 再加入记忆和交叉项；
7. 再加入反馈噪声和同步误差；
8. 最后启用小泄漏和峰值权重；
9. 独立检查EVM、ACLR、输出峰值和系数范数。

常见现象：

| 现象 | 可能原因 | 建议 |
|---|---|---|
| NMSE单调但很慢 | 步长过小 | 逐步提高 `learningRate` |
| NMSE上下振荡 | 步长过大或反馈噪声强 | 降低步长，增加反馈平均 |
| 高阶系数几乎不变 | 高阶尺度过小或样本峰值不足 | 检查特征尺度，增加训练功率覆盖 |
| 单个样点后系数跳变 | 同步毛刺或权重过大 | 检查 `maximumSampleWeight` 和更新投影 |
| 标签NMSE好但ACLR差 | 标签目标或输出限幅不合适 | 使用独立PA级EVM/ACLR验证 |
| 漂移后无法跟踪 | 泄漏过强、步长过小或结构不足 | 减小泄漏、提高步长或增加记忆 |

---

## 15. 适用边界

1. 间接学习假设工作点附近前置逆与后置逆可以共享系数。
2. PA进入不可逆深饱和后，任何LMS逆都无法恢复被压平的信息。
3. 反馈链未去嵌入的频响、IQ失衡和非线性会被误学进DPD。
4. 帧提交模式适合分组通信；无限连续波系统需要明确提交时刻。
5. 逐样点更新减少矩阵存储，不保证比批量岭回归使用更少训练样点。
6. 普通GMP不能独立消除共轭镜像；IQ失衡场景需要将相同更新器扩展到增广GMP特征。
7. MIMO耦合场景需要每路独立更新加跨路特征或先做耦合去嵌入，不能把SISO系数直接复制到所有通道。
