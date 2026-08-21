# 双音信号生成物理原理、参数与使用方法

## 1. 双音信号解决什么问题

Wi-Fi OFDM适合观察宽带EVM和ACLR，但大量子载波同时存在时，很难直接看出某一个非线性阶次产生了什么频谱分量。双音测试只发送两个已知频率，因此奇数阶非线性会在可预测位置产生离散互调分量。工程把双音作为与Wi-Fi并列的激励类型：

```text
WaveGenWifi     → WifiWaveform    → Analysis         → EVM / SNR / ACLR
WaveGenTwoTone  → TwoToneWaveform → TwoToneAnalysis  → IM3 / IM5 / IM7
```

二者都可以直接送入同一个 `PaModel`、`PowerCalibration` 和 `DpdIlc.py`。ILC更新律不需要知道波形来自Wi-Fi还是双音。

## 2. 复基带双音公式

设两个复基带频率按从小到大排列为 $f_1$ 和 $f_2$，幅度为 $A_1$ 和 $A_2$，初相位为 $\phi_1$ 和 $\phi_2$。离散时间双音为

```math
x[n]
=A_1\exp\left(
j\left(2\pi f_1\frac{n}{f_s}+\phi_1\right)
\right)
+A_2\exp\left(
j\left(2\pi f_2\frac{n}{f_s}+\phi_2\right)
\right).
```

其中：

- $f_s$ 是 `sampleRateHz`；
- $n$ 从0递增到 $N-1$；
- $N$ 是 `numSamples`；
- 频率允许为负值，因为这是复基带而不是实数射频波形。

默认使用对称频率

```math
f_1=-2\ \mathrm{MHz},
\qquad
f_2=2\ \mathrm{MHz}.
```

它代表载波中心两侧等间距的两个射频音调。若射频中心频率为 $f_c$，对应的实际射频位置是 $f_c+f_1$ 和 $f_c+f_2$。

## 3. 为什么非线性会产生IM3、IM5和IM7

复包络奇数阶无记忆项可写为

```math
y_p[n]
=a_p x[n]\lvert x[n]\rvert^{p-1},
\qquad
p=1,3,5,7,\ldots
```

把两个复指数代入三阶项 $x\lvert x\rvert^2$ 后，除原来的 $f_1$ 和 $f_2$ 外，还会出现

```math
f_{\mathrm{IM3,L}}=2f_1-f_2,
```

```math
f_{\mathrm{IM3,U}}=2f_2-f_1.
```

五阶和七阶最靠近该阶外侧的双音互调位置分别为

```math
f_{\mathrm{IM5,L}}=3f_1-2f_2,
\qquad
f_{\mathrm{IM5,U}}=3f_2-2f_1,
```

```math
f_{\mathrm{IM7,L}}=4f_1-3f_2,
\qquad
f_{\mathrm{IM7,U}}=4f_2-3f_1.
```

一般的奇数阶 $p$ 可统一写成

```math
f_{\mathrm{IM}p,\mathrm{L}}
=\frac{p+1}{2}f_1-\frac{p-1}{2}f_2,
```

```math
f_{\mathrm{IM}p,\mathrm{U}}
=\frac{p+1}{2}f_2-\frac{p-1}{2}f_1.
```

`TwoToneWaveform.IntermodulationFrequencies(p)` 和 `WaveGenTwoTone.ResolveIntermodulationFrequencies(p)` 使用的就是这组公式。

对默认对称双音，频率位置为：

```text
-14      -10       -6       -2       +2       +6      +10      +14 MHz
 IM7-L    IM5-L     IM3-L     f1       f2      IM3-U    IM5-U    IM7-U
```

图中越靠外的互调阶数越高。实际幅度不一定严格按阶次单调，因为PA记忆、不同非线性系数和DPD抵消项可能相互叠加。

## 4. 两个单音什么时候使用相同功率，什么时候使用不同功率

### 4.1 `toneAmplitudes` 与单音功率的关系

`toneAmplitudes=(A_1,A_2)` 配置的是归一化前的电压幅度比，不是dB功率。对相同阻抗端口，单音平均功率满足

```math
P_1\propto A_1^2,
\qquad
P_2\propto A_2^2.
```

因此两个单音的功率差为

```math
\Delta P_{2-1,\mathrm{dB}}
=
10\log_{10}\frac{P_2}{P_1}
=
20\log_{10}\frac{A_2}{A_1}.
```

生成器最后会用同一个常数缩放整段波形到 `rmsLevel`，所以不会改变 $A_2/A_1$。例如：

| `toneAmplitudes` | 第二音相对第一音功率 | 含义 |
|---|---:|---|
| `(1.0, 1.0)` | 0 dB | 等功率双音 |
| `(1.0, 0.70795)` | 约 -3 dB | 第二音功率减半 |
| `(1.0, 0.5)` | -6.02 dB | 第二音功率为第一音四分之一 |
| `(1.0, 0.1)` | -20 dB | 强音加弱音场景 |

不能把幅度减半误解成降低3 dB；幅度使用20对数，功率使用10对数。

### 4.2 等功率双音的使用场景

等功率条件为

```math
A_1=A_2,
\qquad
P_1=P_2.
```

在固定双音总功率 $P_{\mathrm{total}}=P_1+P_2$ 时，每个单音约承担一半功率。当两个单音初相位相同时，等幅双音还可写成

```math
x(t)
=
2A\cos(\pi\Delta f t)
\exp(j2\pi f_m t),
```

其中

```math
f_m=\frac{f_1+f_2}{2},
\qquad
\Delta f=f_2-f_1.
```

它的包络以 $\Delta f$ 拍频，理想长记录的峰均功率比为2，即约3.01 dB。等功率双音通常用于：

1. 标准PA线性度、IM3、OIP3或IIP3趋势测试；
2. 不同PA、DPD或ILC方法的公平基准比较；
3. 在固定总功率下最大化两个音调的交叉乘积，构造较严格的互调压力；
4. 检查上下互调是否对称，从而发现频率响应、记忆或测量链不对称；
5. 扫描双音间隔，观察电记忆和包络记忆随拍频的变化。

在理想对称、无记忆PA中，上下互调应当对称。因此等功率场景中出现明显的上下侧差异，通常值得检查PA记忆、输入/输出滤波器、IQ不平衡、仪表频响或频率未校准。

### 4.3 不等功率双音的使用场景

不等功率条件为 $P_1\ne P_2$。它不再是传统对称IMD基准，而是用来模拟具有功率层级的真实干扰：

1. 强阻塞信号与弱期望信号共存，观察接收或反馈链去敏感；
2. 不同功率载波聚合、不同资源块或不同制式信号共用PA；
3. 测量强载波对弱载波的交叉调制和增益压缩；
4. 检查反馈ADC、VSA或频谱分析仪是否有足够动态范围同时观测强弱音；
5. 训练或验证能覆盖非均匀业务功率分配的DPD，而不是只覆盖对称实验条件。

设两个单音的复系数为

```math
C_i=A_i\exp(j\phi_i).
```

令奇数阶 $p=2m+1$，只看该阶最外侧双音互调。其复幅度的比例关系为

```math
Y_{p,\mathrm{L}}
\propto
a_p C_1^{m+1}(C_2^*)^m,
```

```math
Y_{p,\mathrm{U}}
\propto
a_p C_2^{m+1}(C_1^*)^m.
```

对应功率近似满足

```math
P_{p,\mathrm{L}}
\propto
|a_p|^2P_1^{m+1}P_2^m,
```

```math
P_{p,\mathrm{U}}
\propto
|a_p|^2P_2^{m+1}P_1^m.
```

所以理想单一阶次模型给出

```math
\frac{P_{p,\mathrm{L}}}{P_{p,\mathrm{U}}}
=
\frac{P_1}{P_2}.
```

强音一侧外部的互调绝对功率更高。以三阶为例：

```math
P_{\mathrm{IM3,L}}\propto P_1^2P_2,
\qquad
P_{\mathrm{IM3,U}}\propto P_2^2P_1.
```

若分别用同侧基波作dBc参考，则理想纯 $p$ 阶模型中两侧近似相同：

```math
\frac{P_{p,\mathrm{L}}}{P_1}
\propto
(P_1P_2)^m,
\qquad
\frac{P_{p,\mathrm{U}}}{P_2}
\propto
(P_1P_2)^m.
```

这意味着不等功率测试必须同时观察：

- `fundamentalLowerDbfs` 与 `fundamentalUpperDbfs`；
- IM的同侧dBc；
- IM的绝对dBFS或dBm。

只看dBc可能隐藏“下侧IM绝对功率明显高于上侧”的事实。`Analysis.CalculateIm3`、`CalculateIm5` 和 `CalculateIm7` 同时返回 `lowerProductDbfs` 与 `upperProductDbfs`，用于观察这种绝对差异。

固定总功率时，乘积 $P_1P_2$ 在 $P_1=P_2$ 时最大。因此对理想多项式PA，等功率双音通常产生最强的归一化交叉互调；改成不等功率后互调变小，并不一定表示PA或DPD变好，而可能只是功率分配不再是最严苛条件。

### 4.4 固定双音间隔时，改变双音位置有什么影响

用中心位置 $f_m$ 和间隔 $\Delta f$ 表示两个音调：

```math
f_1=f_m-\frac{\Delta f}{2},
\qquad
f_2=f_m+\frac{\Delta f}{2}.
```

奇数阶 $p$ 的外侧互调位置可化简为

```math
f_{\mathrm{IM}p,\mathrm{L}}
=
f_m-\frac{p\Delta f}{2},
```

```math
f_{\mathrm{IM}p,\mathrm{U}}
=
f_m+\frac{p\Delta f}{2}.
```

因此保持 $\Delta f$ 不变、把两个音调共同平移 $\delta f$ 时，IM3、IM5和IM7也全部平移相同的 $\delta f$，各谱线之间的相对距离不变。

下面的频谱位置示意图中，两行任意相邻标记的间距都未改变，第二行只是整体右移：

```text
中心0 MHz：  IM7-L  IM5-L  IM3-L   f1    f2   IM3-U  IM5-U  IM7-U
                  <----------- 固定相对间距 ----------->

中心+10 MHz：       IM7-L  IM5-L  IM3-L   f1    f2   IM3-U  IM5-U  IM7-U
                         <----------- 固定相对间距 ----------->
                     整组谱线共同移动 +10 MHz →
```

这张图只描述频率位置。各谱线高度是否保持不变，取决于PA和测量链是否满足后文的平坦、无记忆条件。

共同平移后的输入满足

```math
x_{\delta}(t)
=
x(t)\exp(j2\pi\delta f t),
```

所以

```math
|x_{\delta}(t)|=|x(t)|.
```

拍频周期、包络概率分布、峰均比和固定总功率下的理想耗散功率都不变。对频率平坦、无记忆且具有相位旋转等变性的多项式PA，

```math
y(t)
=
\sum_p a_p x(t)|x(t)|^{p-1},
```

有

```math
y_{\delta}(t)
=
y(t)\exp(j2\pi\delta f t).
```

因此理想模型中IM3、IM5和IM7的dBc不应随双音中心位置改变。这个“不变性”可以作为程序和仪表校验基线。

真实系统通常不是频率平坦、无记忆模型。固定间隔位置扫描能够暴露：

| 观察到的变化 | 更可能的物理原因 |
|---|---|
| 两个基波增益随中心位置一起变化 | PA前后滤波器、耦合器、线缆或仪表幅频响应 |
| 上下基波或上下IM变化不对称 | 幅相不平坦、群时延、PA电记忆或频率相关IQ不平衡 |
| 相同包络下IM仍随位置变化 | GMP延迟支路相位、Wiener滤波器或其他基带记忆响应 |
| 靠近带宽边缘时高阶IM突然恶化或消失 | DPD/DAC/ADC/滤波器带宽不足，或分析频点越界 |
| 镜像与某个IM频点靠近甚至重叠 | Tx/FB IQ镜像随中心位置改变相对关系 |
| 仿真不变而仪表结果变化 | 未建模的RF载频响应、连接网络或仪表校准误差 |

含延迟的GMP支路会对共同频移额外产生与延迟有关的相位。例如 $x[n-m]$ 平移后包含因子

```math
\exp(-j2\pi\delta f m/f_s).
```

不同记忆支路的 $m$ 不同，叠加相位因而随中心位置变化。这就是间隔相同、包络相同，带记忆PA的IM仍可能变化的原因。

推荐位置扫描保持以下条件不变：

1. `toneFrequenciesHz` 的差值不变，只改变平均值；
2. `toneAmplitudes`、相位、记录长度和窗函数不变；
3. 每个位置重新校准到相同实际PA输出dBm；
4. 同时记录基波dBFS、IM dBc、IM绝对dBFS和输出功率；
5. 确认最外侧IM7始终位于Nyquist、仪表带宽和DPD更新带宽内；
6. 固定温度、占空比、等待时间和平均次数。

在本工程复基带仿真中，移动 `toneFrequenciesHz` 测量的是模型已有的基带频响和记忆。实验室若通过改变本振把整组RF双音移动到另一个载频，还会测到PA器件随RF载频变化的特性；只有在PA模型系数或频响也随载频更新时，仿真才能覆盖后一种现象。

### 4.5 等功率和不等功率配置示例

```python
from inc.lib.WaveGenTwoTone import WaveGenTwoTone

equalPowerWaveform = WaveGenTwoTone(
    parameters={
        "toneFrequenciesHz": (-2.0e6, 2.0e6),
        "toneAmplitudes": (1.0, 1.0),
        "width": 0,
    }
).Generate()

strongWeakWaveform = WaveGenTwoTone(
    parameters={
        "toneFrequenciesHz": (-2.0e6, 2.0e6),
        "toneAmplitudes": (1.0, 0.1),
        "width": 0,
    }
).Generate()

shiftedEqualPowerWaveform = WaveGenTwoTone(
    parameters={
        "toneFrequenciesHz": (8.0e6, 12.0e6),
        "toneAmplitudes": (1.0, 1.0),
        "width": 0,
    }
).Generate()
```

第一组用于对称IMD基准；第二组模拟强音比弱音高20 dB；第三组保持4 MHz间隔不变，把双音中心从0 MHz移动到10 MHz，用于位置扫描。

## 5. 有限记录RMS归一化

生成器先在浮点域构造完整有限记录，再计算

```math
x_{\mathrm{rms}}
=\sqrt{
\frac{1}{N}
\sum_{n=0}^{N-1}
\lvert x[n]\rvert^2
}.
```

输出前使用

```math
x_{\mathrm{scaled}}[n]
=x[n]\frac{r_{\mathrm{target}}}{x_{\mathrm{rms}}},
```

其中 $r_{\mathrm{target}}$ 是 `rmsLevel`。默认值为0.5，而不是1，原因是等幅双音的峰值高于RMS。预留峰值空间可以避免16位公开接口在进入PA之前就产生削顶互调。

`rmsLevel` 只是生成器的安全数值幅度，不是最终PA输出dBm。真实工作点由绑定PA的 `PowerCalibration` 闭环建立：

```text
双音原始波形
  → 隐藏输入预设
  → PA
  → 测量实际输出dBm
  → 更新输入预设
  → 误差进入容限
```

## 6. 浮点和定点公开接口

当 `width=0` 时，`samples` 是浮点复包络。当 `width=W>0` 时，I、Q分量按

```math
q_I[n]
=\mathrm{sat}_W\left(
\mathrm{round}\left(2^{W-1}\Re\{x[n]\}\right)
\right),
```

```math
q_Q[n]
=\mathrm{sat}_W\left(
\mathrm{round}\left(2^{W-1}\Im\{x[n]\}\right)
\right)
```

编码为有符号整数码。公开NumPy容器仍是 `complex128`，但实部和虚部都是整数数值。PA、ILC和分析模块在内部计算前统一解码，不要求用户区分两种数组类型。

## 7. 采样率和防混叠

不仅两个基波要位于复基带Nyquist区间，待观察的最高阶互调也必须满足

```math
\left|f_{\mathrm{IM7,L}}\right|<\frac{f_s}{2},
\qquad
\left|f_{\mathrm{IM7,U}}\right|<\frac{f_s}{2}.
```

生成器会同时检查IM3、IM5和IM7。若任一产品越过Nyquist，构造或更新参数时立即报错，因为混叠后的谱线不能再解释成正确阶次。

频域ILC需要生成位于基波外侧的抵消分量。未配置 `ilcBandwidthHz` 时，工程按最远IM7频率自动给出带保护的双边带宽：

```math
B_{\mathrm{ILC}}
=2.2\max\left(
\left|f_{\mathrm{IM7,L}}\right|,
\left|f_{\mathrm{IM7,U}}\right|
\right).
```

该带宽不是双音占用带宽，而是ILC允许更新的频谱范围。

## 8. 参数列表

构造函数为：

```python
WaveGenTwoTone(parameters=None, width=None, **parameterOverrides)
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `sampleRateHz` | `100e6` | 复基带采样率，单位Hz |
| `toneFrequenciesHz` | `(-2e6, 2e6)` | 两个不同的复基带音调频率 |
| `toneAmplitudes` | `(1.0, 1.0)` | 归一化前的相对幅度 |
| `tonePhasesDegrees` | `(0.0, 0.0)` | 两音调初相位，单位度 |
| `numSamples` | `32768` | 有限重复记录的样点数，至少64 |
| `rmsLevel` | `0.5` | 输出边界编码前的整段目标RMS |
| `width` | `16` | 0为浮点，大于0为有符号I/Q位宽 |
| `ilcBandwidthHz` | `None` | 频域ILC带宽；None按IM7自动推导 |

未知键会发出警告并忽略；能识别但物理上非法的值会报错。

## 9. 最小使用示例

```python
from inc.lib.WaveGenTwoTone import WaveGenTwoTone

toneGenerator = WaveGenTwoTone(
    parameters={
        "sampleRateHz": 100.0e6,
        "toneFrequenciesHz": (-2.0e6, 2.0e6),
        "numSamples": 32768,
        "width": 0,
    }
)
toneWaveform = toneGenerator.Generate()

print(toneWaveform.samples.shape)
print(toneWaveform.IntermodulationFrequencies(3))
print(toneWaveform.IntermodulationFrequencies(5))
print(toneWaveform.IntermodulationFrequencies(7))
```

若外部字典需要在运行期间修改：

```python
toneParameters = {
    "sampleRateHz": 100.0e6,
    "toneFrequenciesHz": (-1.0e6, 1.0e6),
    "width": 16,
}

toneGenerator = WaveGenTwoTone(parameters=toneParameters)
firstWaveform = toneGenerator.Generate()

toneParameters["toneFrequenciesHz"] = (-2.0e6, 2.0e6)
secondWaveform = toneGenerator.Generate()
```

`ChainMap` 保留对外部字典的活动视图，因此第二次 `Generate()` 自动使用更新后的频率，不需要重新构造默认参数。

## 10. 适用范围和限制

1. 这是复基带双音，不直接生成实数射频载波。
2. 当前对象生成SISO向量；MIMO双音必须明确每链相位、功率和耦合场景后再扩展，不能简单复制。
3. 双音适合观察离散互调，不替代Wi-Fi调制EVM、频谱掩模或随机业务峰值测试。
4. 若仪表记录存在频偏，准确IM分析前应先用仪表本身或独立同步链校正频率轴。
5. PA热漂移会使迭代间互调缓慢变化；真实仪表测试应固定温度、平均次数和等待时间。
