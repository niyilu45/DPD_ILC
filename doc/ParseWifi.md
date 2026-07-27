# ParseWifi 接收帧解析原理与使用说明

## 1. 模块目标

`inc/lib/ParseWifi.py` 用于从接收到的VHT、HE或EHT复基带波形中恢复 `Analysis` 所需的参考信号和帧元数据。

工程现在保留三条Analysis入口：

1. 已知参考路径：

```python
resultAnalysis = Analysis(referenceSignal, wifiWaveform)
metrics = resultAnalysis.Analyze(receivedSignal)
```

2. 发送波形辅助路径：

```python
resultAnalysis = Analysis(
    receivedSignal,
    transmittedSignal=transmitSignal,
)
metrics = resultAnalysis.Analyze()
```

这条路径不调用 `ParseWifi`。Analysis直接把发送样值作为参考，由 `SigProc.EstimateSignalOverlap` 搜索公共区间。

3. 仅接收帧盲分析路径：

```python
resultAnalysis = Analysis(receivedSignal)
metrics = resultAnalysis.Analyze()
```

只有第三种路径在Analysis内部调用 `ParseWifi.Parse`。Parser恢复参考波形、FFT长度、GI、数据字段位置、活动音调、MCS、空间映射和CSD等信息，然后仍然调用原来的 `SigProc`、`FrameProcess` 和指标计算函数。也就是说，ParseWifi的职责严格限制为“没有任何发送参考时恢复参考信息”；SNR、EVM和ACLR公式没有在Parser中实现。

---

## 2. ParseWifi独立接口的三种解析模式

`ParseWifi.Parse` 的 `receivedSignal` 和可选的 `transmittedSignal` 都可以是 `numpy.ndarray` 或 `WifiWaveform`。调用方始终使用同一个函数和同一组参数名，Parser根据数据类型自动选择内部路径。

> 本节描述的是调用方显式创建 `ParseWifi` 并直接调用 `ParseWifi.Parse(...)` 时的兼容能力，不是 `Analysis` 的发送辅助实现。`Analysis(receivedSignal, transmittedSignal=...)` 始终绕过本节三条Parser分支。

| 模式 | `transmittedSignal` | 元数据来源 | 参考样值来源 | 适用场景 |
|---|---|---|---|---|
| 完全接收解析 | `None` | 接收帧描述字段 | 根据描述字段确定性重生成 | 只有接收采样 |
| 发送样值辅助 | `numpy.ndarray` | 优先解析发送样值中的描述字段 | 用户提供的发送样值 | 已保存DAC或DPD输出波形 |
| 完整发送对象辅助 | `WifiWaveform` | 对象中的完整元数据 | 对象中的发送样值 | 仿真链路内部或完整记录 |

### 2.1 为什么发送波形能够提高准确度

只依赖接收波形时，PA非线性、噪声、CFO、采样频偏和带外滤波都会作用于描述字段。Parser必须联合判断：

- 包起点；
- 接收采样率；
- 描述字段位置；
- 描述比特；
- LDPC校验方程是否成立，或历史CRC是否正确。

提供发送样值后，可以计算发送与接收波形的归一化互相关。包起点由整段波形能量共同决定，不再只依赖两个描述OFDM符号。

如果发送波形记为 $s_m[n]$，第 $m$ 条接收链为 $r_m[n]$，候选时延为 $d$，单链归一化相关功率为：

```math
C_m(d)
=
\frac{
\left|
\sum_{n=0}^{L-1}
r_m[d+n]s_m^*[n]
\right|^2
}{
\left(
\sum_{n=0}^{L-1}|r_m[d+n]|^2
\right)
\left(
\sum_{n=0}^{L-1}|s_m[n]|^2
\right)
}.
```

多链相关得分为：

```math
C(d)
=
\frac{1}{N_{\mathrm{TX}}}
\sum_{m=0}^{N_{\mathrm{TX}}-1}C_m(d).
```

最终包起点为：

```math
\hat d
=
\arg\max_d C(d).
```

先计算每条链的相关功率，再进行平均，可以避免不同PA链公共相位不同而发生复相关相互抵消。

---

## 3. 工程描述字段

### 3.1 为什么需要工程描述字段

本工程的Wi-Fi发生器用于PA和DPD算法研究。它生成符合VHT、HE和EHT字段顺序、OFDM参数、活动音调、MCS星座和空间流结构的复基带激励，但没有实现完整的MAC、标准数据字段BCC/LDPC和标准SIG字段编解码器。本章的短块LDPC只保护工程解析描述。

如果只生成随机SIG激励，接收端无法从随机波形恢复随机种子、MCS和数据符号数。因此 `WaveGenWifi` 在格式相关信令字段中写入一个LDPC保护的工程描述信息：

- VHT使用 `VHT-SIG-A`；
- HE使用 `HE-SIG-A`；
- EHT使用 `U-SIG`。

该描述字段服务于本工程的可复现仿真，不应当解释为IEEE一致性测试所需的bit-exact SIG编码。

### 3.2 55 bit有效载荷

新版描述版本为2。有效载荷由55 bit组成，其中seed由原来的32 bit缩减为10 bit，取值范围为0至1023。

| 字段 | 位数 | 含义 |
|---|---:|---|
| magic word | 12 | 固定同步字 `0xD5B` |
| version | 2 | 新版LDPC描述固定为2 |
| frame format | 2 | VHT、HE或EHT |
| bandwidth | 2 | 20、40、80或160 MHz |
| MCS | 4 | MCS 0至13 |
| guard interval | 2 | 0.4、0.8、1.6或3.2 us |
| data symbol count | 12 | 数据OFDM符号数 |
| TX chain count | 3 | 一至八条物理链 |
| spatial stream count | 3 | 一至八条空间流 |
| spatial mapping | 2 | direct、DFT或custom |
| CSD enabled | 1 | 是否启用循环移位 |
| random seed | 10 | 波形确定性重生成种子，范围0至1023 |
| 合计 | 55 | LDPC系统信息位 |

### 3.3 90 bit短码LDPC

55 bit有效载荷经过系统短码LDPC编码，生成90 bit码字：

```math
\mathbf{c}
=
\begin{bmatrix}
\mathbf{m} \\
\mathbf{p}
\end{bmatrix},
\qquad
\mathbf{m}\in\{0,1\}^{55},
\qquad
\mathbf{p}\in\{0,1\}^{35}.
```

校验矩阵写成：

```math
\mathbf{H}
=
\begin{bmatrix}
\mathbf{A} & \mathbf{B}
\end{bmatrix}.
```

$\mathbf{A}$ 是35乘55的稀疏消息矩阵，每个消息位连接三个校验节点；构造时优先避免两个消息列重复使用同一对校验行，以减少短环。$\mathbf{B}$ 是35乘35的下双对角累加矩阵。编码器递推求解：

```math
\mathbf{B}\mathbf{p}
=
\mathbf{A}\mathbf{m}
\pmod 2,
```

从而保证：

```math
\mathbf{H}\mathbf{c}
=
\mathbf{0}
\pmod 2.
```

码率为：

```math
R
=
\frac{55}{90}
\approx
0.611.
```

接收端使用软输入归一化min-sum译码。设变量节点$v$到校验节点$c$的输入消息为 $L_{v,c}$，校验节点输出近似为：

```math
L_{c,v}
=
0.5
\left(
\prod_{u\ne v}
\mathrm{sign}(L_{u,c})
\right)
\min_{u\ne v}
\left|L_{u,c}\right|.
```

系数0.5用于稳定短码中的低度累加节点。每轮得到后验软值并作硬判决；当全部35个校验方程同时满足时停止。实现仅依赖NumPy，不依赖编译扩展，因此同一代码支持Python 3.9和Python 3.12。

### 3.4 14个分布式导频与跨符号交织

104个物理BPSK音调由以下部分组成：

- 90个LDPC码字音调；
- 14个已知BPSK导频，每个描述OFDM符号放置7个。

每个符号独立使用自己的7个导频估计复增益：

```math
\hat g_s
=
\frac{
\sum_i p_{s,i}^{*}z_{s,i}
}{
\sum_i |p_{s,i}|^2
}.
```

LDPC码字偶数位和奇数位被分散到不同描述符号。这样，某一个OFDM符号受到PA记忆效应、公共相位变化或突发削顶时，不会连续破坏seed或校验字段。

### 3.5 旧版CRC描述兼容

接收端仍可识别历史版本1波形。旧版使用32 bit seed、16 bit CRC-16-CCITT和11 bit保留位。其CRC多项式为：

```math
G(x)=x^{16}+x^{12}+x^5+1.
```

新版发送器不再生成版本1描述，但Parser在LDPC路径失败后会回退到旧版magic、CRC和有限软翻转流程。因此已有保存波形不需要立即转换；新生成波形统一使用10 bit seed和LDPC保护。

### 3.6 Python 3.9和3.12兼容策略

工程没有把描述字段绑定到外部LDPC运行库。现有Python LDPC包要么要求Python 3.10或更高版本，要么还需要SciPy、Numba或编译扩展。为了同时覆盖Python 3.9和Python 3.12，当前实现把编码域步骤集中写在 `inc/lib/Fec.py` 中：

1. 确定性构造稀疏校验矩阵；
2. 系统LDPC编码；
3. 软输入归一化min-sum译码；
4. 校验综合检查。

`ParseWifi.py` 只负责描述载荷装包、导频、交织、均衡和字段语义检查。运行时只使用项目已有的NumPy依赖，没有增加只适配某个Python小版本的二进制包。矩阵在FEC函数内部构造并缓存，不使用模块级可变全局变量。这里的短码用于本工程接收解析描述，不是IEEE 802.11数据字段规定的标准LDPC码字。完整FEC推导和独立函数示例见 [Fec.md](./Fec.md)。

---

## 4. 描述字段的OFDM物理结构

描述比特 $b_i$ 使用BPSK映射：

```math
a_i=1-2b_i.
```

因此：

```math
b_i=0\Rightarrow a_i=+1,
```

```math
b_i=1\Rightarrow a_i=-1.
```

每个20 MHz传统子信道使用负频率26个音调和正频率26个音调。40、80和160 MHz帧在每个bonded 20 MHz子信道上重复发送同一描述信息，提高频率选择性条件下的可靠性。

设传统FFT长度为 $N_{\mathrm{L}}$，频域描述值为 $A[k]$，时域符号为：

```math
x[n]
=
\frac{1}{\sqrt{N_{\mathrm{L}}}}
\sum_{k=0}^{N_{\mathrm{L}}-1}
A[k]\exp\left(
j\frac{2\pi kn}{N_{\mathrm{L}}}
\right).
```

传统GI为0.8 us，循环前缀长度为：

```math
N_{\mathrm{CP,L}}
=
0.8\times 10^{-6}F_s.
```

---

## 5. 不提供发送波形时的解析流程

### 5.1 输入类型和形状

接收输入可以直接是：

```python
parsedFrame = parser.Parse(receivedArray)
```

也可以是：

```python
parsedFrame = parser.Parse(receivedWifiWaveform)
```

输入为 `WifiWaveform` 时，Parser在内部读取 `.samples`，并优先使用对象中的采样率以及可用的custom空间映射矩阵缩小搜索范围；外部接口不发生变化。由于接收对象中的 `.samples` 可能已经包含PA或信道失真，Parser仍通过描述字段恢复独立理想参考，不会简单地把接收样值自身当成零误差参考。

SISO输入是一维复数数组：

```python
receivedSignal.shape == (numSamples,)
```

MIMO输入是样点数乘物理接收链数的矩阵：

```python
receivedSignal.shape == (numSamples, numTransmitAntennas)
```

Parser拒绝空数组、非有限值以及三维或更高维数组。

### 5.2 采样率恢复

接收硬件通常知道自己的采样时钟。用户可以直接配置：

```python
parser = ParseWifi(sampleRateHz=160.0e6)
```

如果 `sampleRateHz=None`，Parser依次尝试类内默认的常见复基带采样率。候选采样率必须形成整数传统FFT长度：

```math
N_{\mathrm{L}}
=
3.2\times 10^{-6}F_s.
```

只有通过新版LDPC校验或历史CRC校验的采样率候选才能继续。

### 5.3 新版LDPC描述字段解调

对候选包起点，Parser删除传统CP并计算单位化FFT：

```math
Y[k]
=
\frac{1}{\sqrt{N_{\mathrm{L}}}}
\sum_{n=0}^{N_{\mathrm{L}}-1}
y[n]\exp\left(
-j\frac{2\pi kn}{N_{\mathrm{L}}}
\right).
```

第 $s$ 个描述OFDM符号中的已知导频记为 $p_{s,i}$，接收值为 $z_{s,i}$。每个符号独立估计复增益：

```math
\hat g_s
=
\frac{
\sum_i p_{s,i}^*z_{s,i}
}{
\sum_i |p_{s,i}|^2
}.
```

独立补偿避免把第一个描述符号的增益和相位强加给第二个符号。设数据音调补偿值为：

```math
q_{s,i}
=
\Re\left\{
\frac{z_{s,i}}{\hat g_s}
\right\}.
```

$q_{s,i}>0$ 倾向于bit 0，$q_{s,i}<0$ 倾向于bit 1，绝对值表示可靠程度。Parser删除14个导频并撤销偶数位、奇数位跨符号交织，得到90个按码字顺序排列的软输入。

初始对数似然近似值为：

```math
L_i^{(0)}
=
\frac{2q_i}{\alpha},
```

其中 $\alpha$ 是非零 $|q_i|$ 的中位数，用于消除整体增益尺度。变量节点和校验节点迭代交换消息，得到后验值：

```math
L_i^{\mathrm{post}}
=
L_i^{(0)}
+
\sum_c L_{c,i}.
```

每轮按 $L_i^{\mathrm{post}}$ 的符号硬判决。只有综合向量满足：

```math
\mathbf{s}
=
\mathbf{H}\hat{\mathbf{c}}
\pmod 2
=
\mathbf{0}
```

时才接受码字，然后继续检查magic、version、格式、带宽、MCS、GI、空间流和seed范围。如果最多60轮仍无法得到零综合，LDPC路径判为失败，不会把不满足校验的比特当作有效配置。

为了进一步排除“校验有效但参数错误”的极小概率事件，Parser用译码参数重新生成完整确定性帧。第 $c$ 个候选在第 $m$ 条物理链上的整帧一致性为：

```math
\gamma_{c,m}
=
\frac{
\left|
\mathbf{x}_{c,m}^{H}\mathbf{y}_{m}
\right|
}{
\sqrt{
\left(\mathbf{x}_{c,m}^{H}\mathbf{x}_{c,m}\right)
\left(\mathbf{y}_{m}^{H}\mathbf{y}_{m}\right)
}
}.
```

其中 $\mathbf{x}_{c,m}$ 是由候选格式、MCS、符号数和随机种子重生成的发送帧，$\mathbf{y}_{m}$ 是接收端仍然可用的重叠样值。二者不要求总长度相等；相关只使用公共有效区间，并用“实际参与相关的样点数/候选帧样点数”的平方根降低不完整候选的评分。逐链相关幅度平均后，Parser选择整帧一致性最高的候选。错误随机种子即使形成一个校验有效码字，也无法同时匹配随机载荷，因此会被排除。采样率低于候选带宽或空间结构不一致时仍直接判为无效。

第 $s$ 个描述符号的导频相关置信度为：

```math
\rho_s
=
\frac{
\left|
\sum_i p_{s,i}^*z_{s,i}
\right|
}{
\sqrt{
\left(\sum_i|p_i|^2\right)
\left(\sum_i|z_i|^2\right)
}
}.
```

两个描述符号的置信度取 $\rho_0$ 和 $\rho_1$ 的平均值。它必须不低于 `minimumParseConfidence`。

如果新版LDPC译码失败，Parser才进入历史版本1兼容路径。兼容路径使用magic word估计公共复增益，并用CRC-16综合值在最低可靠度位中进行有限翻转搜索。旧算法仅用于读取以前保存的32 bit seed波形，新发送器不会再生成这种字段。

### 5.4 包起点细化

CP会使相邻少量采样偏移仍可能得到可判决的FFT结果。Parser找到第一个LDPC有效或历史CRC有效候选后，不立即返回，而是在一个传统CP范围内重新搜索，并选择导频或magic相关值最大的采样位置。

这一步避免把真实包起点前一个采样误判为包起点。

对于本工程生成的PA输出，典型的仅接收调用为：

```python
driveScale = 10.0 ** ((20.0 - 25.0) / 20.0)
paOutput = paModel.Process(driveScale * transmitWaveform.samples)
resultAnalysis = Analysis(paOutput)
metrics = resultAnalysis.Analyze()
```

若PA驱动很深、描述字段已经无法可靠恢复，Analysis应提供发送样值并进入直接发送辅助模式：

```python
resultAnalysis = Analysis(
    paOutput,
    transmittedSignal=transmitWaveform,
)
metrics = resultAnalysis.Analyze()
```

`transmittedSignal` 也可以直接使用 `transmitWaveform.samples`。这条Analysis辅助路径不进入Parser，不读取发送Descriptor，而是用发送/接收互相关找到公共区间并直接把发送样值作为参考。

### 5.5 参考重生成

新版描述字段包含生成器参数和10 bit随机种子。Parser构造新的 `WaveGenWifi` 实例：

```python
referenceWaveform = WaveGenWifi(
    parameters=detectedParameters
).Generate()
```

相同参数和种子会产生相同前导、数据比特、QAM星座、导频、空间映射和CSD。因此Parser可以恢复：

- 理想时域参考；
- `WifiWaveform` 元数据；
- 数据字段切片；
- 数据OFDM符号起点；
- FFT和CP长度；
- 活动、数据和导频音调。

---

## 6. 提供可选发送波形时

本章仅说明调用方直接使用 `ParseWifi.Parse(receivedSignal, transmittedSignal=...)` 的历史兼容接口。新的 `Analysis` 发送辅助模式不会执行本章流程；其公共区间估计已下沉到 `SigProc.EstimateSignalOverlap`。

### 6.1 NumPy发送波形

输入为NumPy数组时：

```python
parsedFrame = ParseWifi().Parse(
    receivedSignal,
    transmittedSignal=transmitSamples,
)
```

处理顺序为：

1. 从发送样值解析描述字段；
2. 恢复Wi-Fi元数据；
3. 保留用户发送样值作为精确参考；
4. 对发送与接收样值执行逐链归一化互相关；
5. 根据相关峰裁剪接收包。

发送和接收总长度不参与相等性判断。Parser使用有符号相关时延寻找两者的最佳公共区间：

```math
n_{\mathrm{tx},0}
=
\max(0,-d),
\qquad
n_{\mathrm{rx},0}
=
\max(0,d),
```

```math
L_{\mathrm{ov}}
=
\min
\left(
L_{\mathrm{tx}}-n_{\mathrm{tx},0},
L_{\mathrm{rx}}-n_{\mathrm{rx},0}
\right).
```

当 $d<0$ 时，接收波形从较长发送波形的中间开始；当 $d>0$ 时，接收捕获在包前带有 $d$ 个前置样点。每一路物理链在长度为 $L_{\mathrm{ov}}$ 的公共区间上分别计算归一化相关，再对各链相关功率求平均。这样支持以下实际流程：

- 原发送数组前后补零，但只提取完整Wi-Fi帧送入PA；
- PA输出或仪表捕获比原发送数组短；
- 发送或接收数组在外部被前后裁剪；
- 接收捕获在包前带有静默区或采集时延。

`EstimateSignalOverlap` 返回接收起点、发送起点、公共区间长度和相关置信度；兼容接口 `EstimatePacketStartFromReference` 仍只返回接收起点和置信度。`maximumPacketOffsetSamples` 只限制接收端包前最多搜索多少样点，不限制发送侧裁剪位置，也不用于比较两个数组的长度。

如果裁剪只删除帧外补零，EVM、SNR和ACLR不受影响。如果裁剪切掉了Wi-Fi帧内部的OFDM样点，Parser仍可利用剩余重叠区间完成对齐，但缺失的帧内信息不能由相关算法恢复，最终EVM会把这些缺失样点体现为误差。

这种方式特别适用于：

- 发送波形已经经过DPD；
- 发送波形不是由当前进程生成，但仍保留工程描述字段；
- 接收描述字段因PA压缩或噪声而难以可靠判决。

### 6.2 WifiWaveform发送对象

输入为 `WifiWaveform` 时：

```python
parsedFrame = ParseWifi().Parse(
    receivedSignal,
    transmittedSignal=wifiWaveform,
)
```

Parser自动检测类型并直接使用：

- `wifiWaveform.samples` 作为发送参考；
- 对象中的FFT、GI、MCS、字段切片和空间映射作为元数据；
- 发送与接收归一化互相关作为包起点估计。

此时不需要描述字段解码，因此它是三种模式中最可靠的方式。

### 6.3 调用方不需要选择模式

下面各调用使用完全相同的函数名和参数名：

```python
parser = ParseWifi()

receiveOnly = parser.Parse(receivedSignal)

objectReceiveOnly = parser.Parse(receivedWifiWaveform)

arrayAssisted = parser.Parse(
    receivedSignal,
    transmittedSignal=transmitSamples,
)

objectAssisted = parser.Parse(
    receivedSignal,
    transmittedSignal=wifiWaveform,
)
```

Parser通过运行时数据类型自动分派。

---

## 7. ParseWifi参数表

所有默认值都定义在 `ParseWifi.__init__` 内部，并通过 `ChainMap` 与外部覆盖值合并。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `sampleRateHz` | `None` | 已知接收采样率；`None`表示自动尝试候选值 |
| `sampleRateCandidatesHz` | 20、40、80、160、320、640 MHz | 自动采样率候选 |
| `maximumPacketOffsetSamples` | `2000` | 接收捕获前允许搜索的最大前置样点数；不限制发送侧裁剪位置 |
| `minimumParseConfidence` | `0.80` | 新版描述导频、历史magic或发送接收相关的最低置信度 |
| `referenceSearchSamples` | `4096` | 发送辅助互相关使用的最大参考样点数 |
| `spatialMappingMatrix` | `None` | custom MIMO映射时由用户补充的矩阵 |
| `width` | `16` | 接收样值与重建参考的I/Q接口位宽；0为浮点，正数表示公开I/Q整数码 |

典型覆盖方式：

```python
parseParameters = {
    "sampleRateHz": 320.0e6,
    "maximumPacketOffsetSamples": 8192,
    "minimumParseConfidence": 0.75,
    "referenceSearchSamples": 8192,
    "width": 16,
}

parser = ParseWifi(parameters=parseParameters)
```

外部字典变化会保持ChainMap的动态覆盖语义。未知键会产生 `UserWarning` 并被忽略，不会阻止接收解析；已识别但非法的采样率、搜索范围或空间映射仍会报错。也可以使用：

```python
parser.UpdateParameters(
    maximumPacketOffsetSamples=16384
)
```

当Parser由 `Analysis` 的盲模式调用时，`Analysis.width` 会自动写入Parser参数，因此接收波形和 `WaveGenWifi` 重建参考使用相同整数码范围。Parser在内部先把码值除以 $2^{width-1}$，再执行相关、描述字段恢复和LDPC译码；`ParsedWifiFrame.receivedSignal` 与 `referenceSignal` 返回调用方时重新编码为公开整数码。直接使用Parser时也可以在 `parseParameters` 中设置 `"width": 0` 或 `"width": 16`。码值算法与饱和边界见 [FixedPoint.md](./FixedPoint.md)。

---

## 8. Analysis典型用法

### 8.1 最简仅接收帧分析

```python
from inc.lib.Analysis import Analysis

resultAnalysis = Analysis(receivedSignal)
metrics = resultAnalysis.Analyze()

print(metrics["snrDb"])
print(metrics["evmDb"])
print(metrics["evmPercent"])
print(metrics["aclrWorstDb"])
```

当采样率不是默认候选值时：

```python
resultAnalysis = Analysis(
    receivedSignal,
    parseParameters={"sampleRateHz": 30.0e6},
)
metrics = resultAnalysis.Analyze()
```

### 8.2 Analysis的NumPy发送样值辅助

```python
resultAnalysis = Analysis(
    receivedSignal,
    transmittedSignal=transmitSamples,
)
metrics = resultAnalysis.Analyze()
```

`transmitSamples` 可以是一维SISO数组，也可以是 `samples × chains` 的MIMO矩阵。

这里Analysis不会调用 `ParseWifi.Parse`。它直接搜索两路波形公共区间，因此发送样值可以被裁剪、前后补零，也不要求包含Descriptor。纯NumPy输入没有帧元数据，结果采用波形域EVM/SNR；需要ACLR时同时传入 `sampleRateHz` 和 `channelBandwidthHz`。

已有程序如果通过 `parseParameters` 提供接收机采样率，也不需要修改调用结构：

```python
resultAnalysis = Analysis(
    receivedSignal,
    transmittedSignal=transmitSamples,
    parseParameters={
        "sampleRateHz": 320.0e6,
        "channelBandwidthHz": 80.0e6,
    },
)
metrics = resultAnalysis.Analyze()
```

此处 `parseParameters` 只是采样率/带宽的兼容入口，Analysis仍然不调用
`ParseWifi.Parse`。Parser专用的其他键会警告后忽略；显式
`sampleRateHz=` 和 `channelBandwidthHz=` 具有更高优先级。

### 8.3 Analysis的WifiWaveform发送对象辅助

```python
wifiGenerator = WaveGenWifi(
    frameFormat="EHT",
    bandwidthMhz=80,
    mcs=11,
    sampleRateHz=320.0e6,
)
wifiWaveform = wifiGenerator.Generate()
receivedSignal = paModel.Process(wifiWaveform.samples)

resultAnalysis = Analysis(
    receivedSignal,
    transmittedSignal=wifiWaveform,
)
metrics = resultAnalysis.Analyze()
```

这里仍然不调用Parser。Analysis读取 `wifiWaveform.samples` 作为真实参考，并复用对象中已有的字段边界、FFT/GI和空间映射元数据计算严格Wi-Fi子载波EVM。

### 8.4 原有已知参考方式继续保留

```python
resultAnalysis = Analysis(
    referenceSignal,
    wifiWaveform,
)
metrics = resultAnalysis.Analyze(receivedSignal)
```

显式参考路径的 `Analyze` 仍然要求传入 `measuredSignal`。发送辅助路径和Parser构造的盲接收路径都在构造时保存了测量样值，因此可以使用零参数 `Analyze()`。

### 8.5 独立使用ParseWifi

```python
from inc.lib.ParseWifi import ParseWifi

parser = ParseWifi(
    sampleRateHz=80.0e6,
    maximumPacketOffsetSamples=2048,
)
parsedFrame = parser.Parse(
    receivedSignal,
    transmittedSignal=optionalTransmitSignal,
)

print(parsedFrame.packetStartSample)
print(parsedFrame.parseConfidence)
print(parsedFrame.detectedParameters)

resultAnalysis = Analysis(
    parsedFrame.referenceSignal,
    parsedFrame.waveform,
)
metrics = resultAnalysis.Analyze(
    parsedFrame.receivedSignal
)
```

如果既有脚本把工程的 `inc` 目录加入了 `sys.path`，也可以使用 `from lib.ParseWifi import ParseWifi`。模块内部会自动选择与当前包层级匹配的 `utils` 导入路径，避免相对导入越过顶层包。新代码仍推荐使用完整的 `inc.lib.ParseWifi` 包名。

---

## 9. ParsedWifiFrame字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `receivedSignal` | `numpy.ndarray` | 从接收包起点提取的可用重叠波形；允许短于发送参考 |
| `referenceSignal` | `numpy.ndarray` | 重生成或用户提供的发送参考 |
| `waveform` | `WifiWaveform` | Analysis和FrameProcess需要的完整元数据 |
| `packetStartSample` | `int` | 原始接收捕获中的包起点 |
| `parseConfidence` | `float` | 零至一的相关置信度 |
| `detectedParameters` | 只读映射 | 格式、带宽、MCS、GI、空间流和采样率等 |

通过Analysis取得该对象：

```python
parsedFrame = resultAnalysis.GetParsedWifiFrame()
```

显式参考路径返回 `None`。

---

## 10. 自定义MIMO空间映射

描述字段可以指出映射类型为 `custom`，但104比特空间不足以携带任意复矩阵。没有 `WifiWaveform` 对象时，需要提供：

```python
parser = ParseWifi(
    spatialMappingMatrix=customMappingMatrix
)
```

如果可选发送输入本身就是 `WifiWaveform`，Parser直接读取对象中的矩阵，不需要额外配置。

NumPy发送样值虽然有助于同步，但不能唯一确定任意自定义空间映射矩阵。仅靠接收数据估计任意MIMO信道和空间映射属于信道估计问题，不应通过类型猜测隐式完成。

---

## 11. 适用边界

### 11.1 当前支持

- 本工程 `WaveGenWifi` 生成的VHT、HE和EHT帧；
- 20、40、80和160 MHz；
- 各格式支持的全部MCS；
- SISO和最多八条物理链；
- direct和DFT空间映射；
- 提供矩阵后的custom空间映射；
- 捕获开头存在有限前置样点；
- PA非线性、公共复增益和适度噪声。

### 11.2 当前不等价于商业Wi-Fi芯片解码器

外部标准空口抓包通常使用标准L-SIG、VHT-SIG、HE-SIG和U-SIG编码，还会包含：

- 信道编码和交织；
- 包检测和AGC；
- 多径信道估计与均衡；
- 导频相位跟踪；
- 相位噪声补偿；
- MAC长度和聚合结构。

本Parser的描述字段是工程仿真协议。没有该字段且没有 `WifiWaveform` 元数据时，不能承诺自动解析任意商业设备抓包。

### 11.3 EVM含义

无发送参考时，Parser根据描述字段中的种子重建原始随机QAM数据，因此EVM仍是数据辅助EVM，不是把接收星座硬判决后再当作真值的decision-directed EVM。

这种方式避免了高噪声时硬判决错误被错误地当成零误差参考。

---

## 12. 常见错误

### 12.1 无法找到描述字段

检查：

- 波形是否由更新后的 `WaveGenWifi` 生成；
- 捕获是否包含完整格式信令字段；
- 接收采样率是否在候选列表内；
- 前置样点是否超过 `maximumPacketOffsetSamples`；
- PA是否已经把信令字段严重削顶。

可以提供发送波形提高可靠性。

### 12.2 发送和接收链数不一致

NumPy发送数组和接收数组必须都为一维，或具有相同的矩阵列数。列代表物理PA链，不能把空间流数直接当成矩阵列数。

### 12.3 互相关置信度不足

可检查：

- 发送与接收是否属于同一帧；
- 是否存在未补偿的极大采样频偏；
- `referenceSearchSamples` 是否过短；
- `minimumParseConfidence` 是否高于当前测量条件允许值。

不建议为了通过解析而把阈值降到很低，因为错误帧配对会使后续SNR和EVM失去物理意义。
