# 双音IM3、IM5、IM7分析原理与ILC性能比较

## 1. 指标定义

`TwoToneAnalysis` 不计算Wi-Fi EVM，也不解析帧头。它只消费 `TwoToneWaveform` 中的精确频率元数据，并对PA输出计算：

- 两个基波电平；
- IM3下侧、上侧和较差侧；
- IM5下侧、上侧和较差侧；
- IM7下侧、上侧和较差侧；
- 三个阶次中最大的剩余互调；
- PA稳态平均输出功率dBm。

所有结果由 `Analyze(measuredSignal)` 以普通Python字典返回。其中
`outputPowerDbm` 是送入分析器的模拟PA输出参考面功率；它与IM3、IM5、
IM7在同一次调用中产生，不需要调用方再写一套RMS功率计算。

主分析类 `Analysis` 还提供 `AnalyzeTwoTone`、`CalculateIm3`、`CalculateIm5` 和 `CalculateIm7` 静态入口。这些入口内部委托给 `TwoToneAnalysis`，不会复制另一套频谱算法，也不会把双音逻辑混入Wi-Fi的实例 `Analyze()` 路径。静态入口既接受原有的 `TwoToneWaveform`，也接受NumPy数组或Python列表；原始样值模式必须同时给出物理 `sampleRateHz` 与两个 `toneFrequenciesHz`，因为仅凭一段复样值无法可靠推断音调的真实频率标尺。

## 2. 为什么不用最近FFT格点

若音调频率不是有限记录FFT分辨率的整数倍，把功率直接读取到最近FFT格点会产生栅栏误差。工程使用已知物理频率的加窗复投影。设去掉首尾暂态后的记录为 $y[n]$，窗函数为 $w[n]$，需要测量的频率为 $f$，则复幅度估计为

```math
\hat A(f)
=
\frac{
\sum_{n=0}^{N-1}
w[n]y[n]\exp\left(-j2\pi f n/f_s\right)
}{
\sum_{n=0}^{N-1}w[n]
}.
```

分母是窗的相干增益。若输入正好是 $A\exp(j2\pi fn/f_s)$，并且其他谱线距离足够远，则 $\hat A(f)$ 接近 $A$，不要求 $f$ 落在FFT整数格点上。

默认 `windowName="hann"`。Hann窗降低强基波泄漏到弱互调频点的旁瓣。若信号严格相干且记录包含整数周期，也可以使用 `rectangular`。

## 3. 为什么要去掉首尾暂态

Wiener PA的线性记忆滤波器和GMP的延迟项在记录开头没有完整历史。若把开头暂态作为稳态谱的一部分，会出现并非连续双音非线性产生的宽带泄漏。分析器默认从两端各去掉256点：

```math
y_{\mathrm{steady}}[n]
=y[n+N_{\mathrm{settle}}],
```

有效长度为

```math
N_{\mathrm{effective}}
=N-2N_{\mathrm{settle}}.
```

`settlingSamples` 必须保证至少留下64个分析样点。

## 4. dBc的计算

设两个基波的投影功率为

```math
P_1=\left|\hat A(f_1)\right|^2,
\qquad
P_2=\left|\hat A(f_2)\right|^2.
```

下侧互调相对于下侧基波，上侧互调相对于上侧基波。例如IM3：

```math
\mathrm{IM3}_{\mathrm{L,dBc}}
=10\log_{10}
\left(
\frac{
\left|\hat A(2f_1-f_2)\right|^2
}{
P_1
}
\right),
```

```math
\mathrm{IM3}_{\mathrm{U,dBc}}
=10\log_{10}
\left(
\frac{
\left|\hat A(2f_2-f_1)\right|^2
}{
P_2
}
\right).
```

IM5和IM7只需把频率替换为对应产品位置。一个阶次的较差侧定义为

```math
\mathrm{IM}p_{\mathrm{worst,dBc}}
=\max\left(
\mathrm{IM}p_{\mathrm{L,dBc}},
\mathrm{IM}p_{\mathrm{U,dBc}}
\right).
```

dBc通常为负数。数值越负，互调相对于基波越小，性能越好。例如从-30 dBc降到-45 dBc表示改善15 dB。

### 4.1 不等功率双音为什么还要看绝对dBFS

等功率双音中，两个基波相同，左右互调的dBc可以直接对称比较。不等功率双音中，本工程仍采用“同侧互调除以同侧基波”的定义：下侧互调相对于下侧基波，上侧互调相对于上侧基波。此时只看dBc可能掩盖两个互调产物的绝对电平差异，因此还应计算

```math
L_{\mathrm{IM,L,dBFS}}
=
L_{f_1,\mathrm{dBFS}}
+
L_{\mathrm{IM,L,dBc}},
```

```math
L_{\mathrm{IM,U,dBFS}}
=
L_{f_2,\mathrm{dBFS}}
+
L_{\mathrm{IM,U,dBc}}.
```

`Analysis.CalculateIm3`、`CalculateIm5` 和 `CalculateIm7` 会同时返回 `lowerProductDbfs`、`upperProductDbfs` 与两侧dBc。因此推荐：

- 对称PA、IP3或不同DPD方法的公平基准采用等功率双音；
- 强阻塞加弱有用信号、非对称载波聚合或交叉调制场景采用不等功率双音；
- 不等功率测试同时报告基波dBFS、互调dBc和互调绝对dBFS；
- 扫描功率比例时固定总输入或实际PA输出功率，避免把总功率变化误判为功率比例效应。

双音功率比例和固定间隔移动位置的完整推导见 [WaveGenTwoTone.md第4节](./WaveGenTwoTone.md#4-两个单音什么时候使用相同功率什么时候使用不同功率)。

三种阶次的综合选择值为

```math
I_{\mathrm{worst}}
=\max\left(
\mathrm{IM3}_{\mathrm{worst}},
\mathrm{IM5}_{\mathrm{worst}},
\mathrm{IM7}_{\mathrm{worst}}
\right).
```

`AnalyzeIlcHistory` 在每一轮真实PA输出上独立计算这些值，并选择 $I_{\mathrm{worst}}$ 最小的实测轮。原始 `DpdIlc.py` 仍然只保存波形和MSE，不嵌入任何IM指标。

## 5. 模拟输出功率、参考面和定点换算

### 5.1 功率测量参考面

`outputPowerDbm` 测量的是 `measuredSignal` 所在参考面的功率。推荐把PA的
直接输出送入该参数，此时它就是模拟PA输出端口功率；如果调用方先对波形
加入额外线性增益、PA后耦合或接收链增益，结果自然变成这些模块之后的
观测参考面功率。分析器不会把PA输入功率当成输出功率，也不会简单复制
`PowerCalibration` 的目标值。

分析前先去掉首尾各 `settlingSamples` 个暂态样点，再根据
`activePowerThresholdDb` 找到活动样点，并用
`activeGapToleranceSamples` 闭合活动区中的短过零间隙。长补零和占空比空闲
不会拉低活动段功率。令最终活动样点集合为 $\mathcal A$，则归一化复包络
RMS为

```math
y_{\mathrm{rms}}
=\sqrt{
\frac{1}{|\mathcal A|}
\sum_{n\in\mathcal A}
\left|y[n]\right|^2
}.
```

工程约定归一化PA输出RMS为1时对应 `maximumOutputPowerDbm`。设端口阻抗
为 $R$，该满量程功率对应的RMS电压为

```math
V_{\mathrm{FS}}
=
\sqrt{
R\times 10^{-3}
\times 10^{P_{\max,\mathrm{dBm}}/10}
}.
```

活动信号的模拟RMS电压为 $V_{\mathrm{rms}}=y_{\mathrm{rms}}V_{\mathrm{FS}}$，
所以报告功率为

```math
P_{\mathrm{out,dBm}}
=P_{\max,\mathrm{dBm}}
+20\log_{10}\left(y_{\mathrm{rms}}\right).
```

这里的 `loadResistanceOhm` 定义模拟端口阻抗，而
`maximumOutputPowerDbm` 定义归一化满量程的绝对功率。只要所有模块采用
相同的两个参考值，双音分析、普通Analysis和功率闭环就处于同一模拟功率
参考面。

### 5.2 定点码为什么不能直接当成浮点或伏特

当 `width=W>0` 时，公开NumPy数组的实部和虚部虽然存储在
`complex128` 中，但数值是有符号整数码，不是归一化小数，也不是伏特。
设PA输出每个I/Q分量的物理码轨为 $F_{\mathrm{out}}$，分析器必须先执行

```math
y[n]
=
F_{\mathrm{out}}
\frac{c_I[n]+j c_Q[n]}{2^{W-1}},
```

再计算活动区RMS和dBm。频谱中的dBFS使用 $y/F_{\mathrm{out}}$，所以扩大观测量程不会改变同一整数码相对于码轨的电平。若把码值 $c[n]$ 直接当作浮点复包络，功率会额外
增加

```math
\Delta P_{\mathrm{dB}}
=20\log_{10}\left(\frac{2^{W-1}}{F_{\mathrm{out}}}\right).
```

16位旧FS1接口的纯缩放误差为约90.31 dB，因而本应约为20 dBm的波形曾可能被错误报告成110至112 dBm。当前PA/Channel定点输出默认FS2，误差项为约84.29 dB；无论具体量程为何，这都不是PA产生了超高功率，而是把整数码误当成模拟幅度或使用了错误输出标尺。

原始NumPy/list模式可以对明显超出归一化范围的整数码自动识别工程默认
16位。这个启发式不用于带 `TwoToneWaveform` 的元数据模式：该模式在省略
`width` 时继承发送波形位宽。如果发送参考是浮点而接收仪表导出16位码，
必须显式传入 `width=16`；否则接收码会按发送端的 `width=0` 解释，正好
产生上述约90.31 dB偏差。

例如 `maximumOutputPowerDbm=25.0`、目标输出为20 dBm时，正确的物理
活动区RMS应接近

```math
y_{\mathrm{rms,target}}
=10^{(20-25)/20}
\approx0.5623.
```

在默认FS2的16位PA输出接口中，这相当于约
$0.5623/2\times32768\approx9213$ 的复包络码RMS。分析器使用 `width=16, outputFullScaleAmplitude=2.0` 解码后仍得到约20 dBm；若错误沿用FS1，会得到约14 dBm；若使用 `width=0` 又把9213直接当成物理RMS，则会多出约84.29 dB。

工程的标尺约定是：WaveGen/DPD/DAC输入默认FS1，`PaModel` 和 `Channel` 输出默认FS2，接近25 dBm且高PAPR峰值可能越过FS2时可把PA、Channel和分析器统一改为FS4。`TwoToneAnalysis` 与普通 `Analysis` 必须匹配实际输出标尺；只修改其中一端会造成6.02 dB（FS1与FS2）或更大的幅度/功率错误。

最终方法对比前，Benchmark会把每种ILC选择出的输入重新交给闭环 `PowerCalibration`。校准器只调整PA输入，直到实际输出功率进入容限，不在PA输出端做常数缩放。因此各方法的IM3、IM5和IM7是在相同真实输出dBm下比较。

## 6. 为什么一种ILC可能改善IM3却恶化IM7

有限迭代、峰值约束和模型阶数会形成权衡：

1. IM3通常是最强互调，时域MSE和综合最大互调都更容易被IM3主导。
2. 用三阶形状抵消IM3时，预失真波形经过PA的五阶和七阶项会继续混频。
3. 参数域MP ILC若阶数、记忆深度或正则化不合适，可能降低IM3和IM5，但放大较弱的IM7。
4. 定点量化底噪会先限制最弱的IM7，导致IM7不再随MSE平滑变化。
5. 频域ILC若投影带宽只覆盖到IM3，就无法合成IM5和IM7所需的带外抵消分量。

因此报告必须同时列出三种阶次，不能只用一个“总失真”数字宣布某种方法全面更好。

## 7. 参数列表

构造函数为：

```python
TwoToneAnalysis(waveform, parameters=None, width=None, outputFullScaleAmplitude=None, **parameterOverrides)
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `windowName` | `"hann"` | 精确频率投影前的窗；支持hann或rectangular |
| `settlingSamples` | `256` | 记录首尾各丢弃的暂态样点数 |
| `minimumSpectralPower` | `1e-30` | 对数计算功率下限，避免负无穷 |
| `maximumOutputPowerDbm` | `25.0` | 归一化输出RMS为1时的PA功率 |
| `loadResistanceOhm` | `50.0` | 与工程其他功率模块一致的端口阻抗 |
| `activePowerThresholdDb` | `-60.0` | 相对活动段峰值的功率检测门限，低于门限的长补零或空闲不进入模拟功率RMS |
| `activeGapToleranceSamples` | `16` | 活动区内允许闭合的短过零间隙样点数 |
| `width` | 省略时自动识别或继承元数据 | 描述 `measuredSignal` 的边界；典型16位整数码可从浮点发送元数据中自动识别，0为浮点，大于0为公开整数I/Q码 |
| `outputFullScaleAmplitude` | `1.0` | 接收/PA输出整数码轨表示的物理I/Q分量幅度；兼容旧FS1采集，当前PA/Channel默认输出应显式设为2.0 |

## 8. PA输出分析示例

```python
from inc.lib.PaModel import PaModel
from inc.lib.TwoToneAnalysis import TwoToneAnalysis
from inc.lib.WaveGenTwoTone import WaveGenTwoTone
from inc.utils.SigProc import PowerCalibration

toneWaveform = WaveGenTwoTone(
    parameters={
        "sampleRateHz": 100.0e6,
        "toneFrequenciesHz": (-2.0e6, 2.0e6),
        "numSamples": 32768,
        "width": 0,
    }
).Generate()

paModel = PaModel(
    parameters={
        "modelName": "wiener",
        "width": 0,
    }
)

powerCalibration = PowerCalibration(
    paModel=paModel,
    parameters={
        "outputPowerDbm": 20.0,
        "maximumOutputPowerDbm": 25.0,
        "width": 0,
    },
)

referenceSignal = powerCalibration.Calibrate(toneWaveform.samples)
paOutput = powerCalibration.GetLastPaOutput()

toneAnalysis = TwoToneAnalysis(
    toneWaveform,
    parameters={
        "maximumOutputPowerDbm": 25.0,
        "width": 0,
        "outputFullScaleAmplitude": paModel.outputFullScaleAmplitude,
    },
)
metrics = toneAnalysis.Analyze(paOutput)

print(metrics["im3WorstDbc"])
print(metrics["im5WorstDbc"])
print(metrics["im7WorstDbc"])
print(f"PA output power: {metrics['outputPowerDbm']:.2f} dBm")
assert abs(metrics["outputPowerDbm"] - 20.0) <= 0.25
```

### 8.1 通过主Analysis类分别读取IM3、IM5和IM7

当上层程序希望统一从 `Analysis` 导入测量接口时，可以直接传入PA输出和 `TwoToneWaveform`：

```python
from inc.lib.Analysis import Analysis

toneParameters = {
    "maximumOutputPowerDbm": 25.0,
    "width": 0,
    "outputFullScaleAmplitude": paModel.outputFullScaleAmplitude,
}
allMetrics = Analysis.AnalyzeTwoTone(
    paOutput,
    toneWaveform,
    parameters=toneParameters,
)
im3Metrics = Analysis.CalculateIm3(
    paOutput, toneWaveform, parameters=toneParameters
)
im5Metrics = Analysis.CalculateIm5(
    paOutput, toneWaveform, parameters=toneParameters
)
im7Metrics = Analysis.CalculateIm7(
    paOutput, toneWaveform, parameters=toneParameters
)

print(allMetrics["worstIntermodulationDbc"])
print(allMetrics["outputPowerDbm"])
print(allMetrics["im3LowerDbc"], allMetrics["im3UpperDbc"])
print(allMetrics["im5LowerDbc"], allMetrics["im5UpperDbc"])
print(allMetrics["im7LowerDbc"], allMetrics["im7UpperDbc"])
print(im3Metrics["worstDbc"], im3Metrics["lowerProductDbfs"])
print(im5Metrics["worstDbc"], im5Metrics["lowerProductDbfs"])
print(im7Metrics["worstDbc"], im7Metrics["lowerProductDbfs"])
```

每个单阶字典包含：

| 字段 | 含义 |
|---|---|
| `nonlinearOrder` | 3、5或7 |
| `lowerFrequencyHz`, `upperFrequencyHz` | 该阶下侧和上侧互调物理频率 |
| `lowerDbc`, `upperDbc`, `worstDbc` | 同侧基波归一化后的互调指标 |
| `lowerProductDbfs`, `upperProductDbfs` | 两个互调产物的绝对归一化电平 |
| `outputPowerDbm` | 与完整分析相同的模拟PA输出参考面功率 |

一次需要全部阶次时优先调用 `AnalyzeTwoTone`，只执行一轮投影。三个专用方法更适合只验收某一阶指标或把单阶结果送入自动测试接口。

### 8.2 直接使用NumPy数组或Python列表

若发送端来自仪表、真实芯片或其他仿真器，不必先构造 `TwoToneWaveform`。可直接提供PA输出样值；也可以提供原始发送样值作为第二个位置参数。原始数组本身只用于建立样点数和定点接口元数据，IM测量仍由明确给出的频率完成精确投影，不会把发送数组重新生成为“理想参考”。

```python
from inc.lib.Analysis import Analysis


rawMetrics = Analysis.AnalyzeTwoTone(
    paOutput.tolist(),
    transmittedTwoToneSamples.tolist(),
    sampleRateHz=100.0e6,
    toneFrequenciesHz=(-2.0e6, 2.0e6),
    parameters={
        "settlingSamples": 256,
        "maximumOutputPowerDbm": 25.0,
        "width": 0,
    },
)

# A separate transmit array is optional when only spectrum metrics are needed.
standaloneMetrics = Analysis.AnalyzeTwoTone(
    paOutput,
    sampleRateHz=100.0e6,
    toneFrequenciesHz=(-2.0e6, 2.0e6),
    parameters={"width": 0},
)
im3Metrics = Analysis.CalculateIm3(
    paOutput.tolist(),
    sampleRateHz=100.0e6,
    toneFrequenciesHz=(-2.0e6, 2.0e6),
    parameters={"width": 0},
)

print(rawMetrics["im3LowerDbc"], rawMetrics["im3UpperDbc"])
print(rawMetrics["im5LowerDbc"], rawMetrics["im5UpperDbc"])
print(rawMetrics["im7LowerDbc"], rawMetrics["im7UpperDbc"])
print(rawMetrics["outputPowerDbm"])
```

原始样值模式的规则如下：

- `sampleRateHz` 和 `toneFrequenciesHz` 是必填物理量；缺少任一项会报错，而不是猜测错误频率；
- `AnalyzeTwoTone` 对IM3、IM5和IM7均同时返回 `LowerDbc` 与 `UpperDbc`；每一阶的 `WorstDbc` 只是两侧中的较差者，不能替代两侧原始结果；
- 两个基波必须不同，且IM3、IM5和IM7理论位置都必须位于复Nyquist范围内；
- 原始发送数组和PA输出必须具有相同长度；
- 原始NumPy/list模式省略 `width` 时会检查全部I/Q分量：若它们都是整数码形态且至少一个分量的绝对值大于1，则识别为工程默认16位；其他情况按浮点 `width=0` 处理。为了避免幅度很小的整数码、整数值浮点测试信号等歧义，生产测试仍推荐显式配置位宽；8、12、24位等非默认格式必须明确提供；
- `TwoToneWaveform` 模式仍是最完整的频率元数据调用方式。若同时提供它和 `sampleRateHz` 或 `toneFrequenciesHz`，物理频率参数必须一致；显式 `width` 只描述接收 `measuredSignal`，允许与发送 `TwoToneWaveform.width` 不同。若发送元数据为浮点且接收样值呈现典型16位整数码形态，省略 `width` 时同样自动识别16位；其他非零发送位宽则默认继承。

16位原始码的推荐写法如下。`fixedPaOutput` 必须是模块公开返回的整数码，
不能预先除以32768后仍声明 `width=16`，也不能保持码值不变却声明
`width=0`：

```python
fixedMetrics = Analysis.AnalyzeTwoTone(
    fixedPaOutput.tolist(),
    sampleRateHz=100.0e6,
    toneFrequenciesHz=(-2.0e6, 2.0e6),
    parameters={
        "settlingSamples": 256,
        "maximumOutputPowerDbm": 25.0,
        "activePowerThresholdDb": -60.0,
        "activeGapToleranceSamples": 16,
        "width": 16,
        "outputFullScaleAmplitude": 2.0,
    },
)

print(f"PA output power: {fixedMetrics['outputPowerDbm']:.2f} dBm")
```

## 9. 单一ILC逐轮分析示例

```python
from inc.lib.DpdIlc import ILCConfig, RunFrequencyDomainIlc

ilcResult = RunFrequencyDomainIlc(
    referenceSignal,
    paModel,
    toneWaveform.sampleRateHz,
    toneWaveform.ilcBandwidthHz,
    ILCConfig(
        numIterations=8,
        learningRate=0.15,
        maxAmplitude=1.5,
    ),
)

analyzedResult = toneAnalysis.AnalyzeIlcHistory(ilcResult.history)
toneAnalysis.SaveIlcHistory(
    analyzedResult,
    "results/two_tone_single_method",
    "frequency_domain_ilc",
)
```

每轮文件同时保存原生NMSE和IM3、IM5、IM7。这样可以观察“MSE继续下降但某个高阶互调开始反弹”的情况。

## 10. 全部ILC方法比较

```powershell
python tests/BenchMark.py --two-tone --sample-rate-hz 100000000 --tone-lower-hz -2000000 --tone-upper-hz 2000000 --tone-samples 32768 --output-power-dbm 20 --iterations 10 --pa wiener --output-dir results/two_tone_ilc_benchmark
```

该入口比较所有适用于SISO双音的更新律：

- Scalar P ILC；
- Complex-gain ILC；
- FIR ILC；
- Frequency-domain ILC；
- Directional Gauss-Newton ILC；
- Parameter-domain MP ILC；
- Augmented IQ ILC。

输出包括：

```text
all_ilc_two_tone_metrics.csv
all_ilc_two_tone_metrics.json
all_ilc_two_tone_imd.png
histories/*.csv
histories/*.json
```

PNG把所有方法的IM3、IM5和IM7较差侧画在同一张分组柱状图上。

## 11. 参考数值和解释

以下数值来自浮点、4096点、20 dBm、Wiener PA、每种方法2轮的快速结构检查，只用于说明字段和趋势，不作为固定算法指标：

| 方法 | IM3较差侧/dBc | IM5较差侧/dBc | IM7较差侧/dBc |
|---|---:|---:|---:|
| PA baseline | -33.25 | -45.72 | -68.77 |
| Scalar P ILC | -34.10 | -46.54 | -68.00 |
| Complex-gain ILC | -34.56 | -46.98 | -67.61 |
| FIR ILC | -34.59 | -46.95 | -67.62 |
| Frequency-domain ILC | -34.56 | -46.98 | -67.60 |
| Directional Gauss-Newton ILC | -33.30 | -45.77 | -68.72 |
| Parameter-domain MP ILC | -34.82 | -47.35 | -66.09 |
| Augmented IQ ILC | -33.77 | -46.60 | -67.56 |

这组快速结果说明：两轮时IM3和IM5已经开始改善，但若优化主要由较强的IM3主导，较弱IM7可能暂时反弹。正式结论必须增加迭代数、改变输出功率、同时检查输入峰值，并在相同配置下重复比较。
