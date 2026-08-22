# 双音IM3、IM5、IM7分析原理与ILC性能比较

## 1. 指标定义

`TwoToneAnalysis` 不计算Wi-Fi EVM，也不解析帧头。它只消费 `TwoToneWaveform` 中的精确频率元数据，并对PA输出计算：

- 两个基波电平；
- IM3下侧、上侧和较差侧；
- IM5下侧、上侧和较差侧；
- IM7下侧、上侧和较差侧；
- 三个阶次中最大的剩余互调；
- PA稳态平均输出功率dBm。

所有结果由 `Analyze(measuredSignal)` 以普通Python字典返回。

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

## 5. 输出功率

稳态记录RMS为

```math
y_{\mathrm{rms}}
=\sqrt{
\frac{1}{N_{\mathrm{effective}}}
\sum_n
\left|y_{\mathrm{steady}}[n]\right|^2
}.
```

工程约定归一化PA输出RMS为1时对应 `maximumOutputPowerDbm`。因此报告功率为

```math
P_{\mathrm{out,dBm}}
=P_{\max,\mathrm{dBm}}
+20\log_{10}\left(y_{\mathrm{rms}}\right).
```

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
TwoToneAnalysis(waveform, parameters=None, width=None, **parameterOverrides)
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `windowName` | `"hann"` | 精确频率投影前的窗；支持hann或rectangular |
| `settlingSamples` | `256` | 记录首尾各丢弃的暂态样点数 |
| `minimumSpectralPower` | `1e-30` | 对数计算功率下限，避免负无穷 |
| `maximumOutputPowerDbm` | `25.0` | 归一化输出RMS为1时的PA功率 |
| `loadResistanceOhm` | `50.0` | 与工程其他功率模块一致的端口阻抗 |
| `width` | 继承波形 | 0为浮点，大于0为公开整数I/Q码 |

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
    },
)
metrics = toneAnalysis.Analyze(paOutput)

print(metrics["im3WorstDbc"])
print(metrics["im5WorstDbc"])
print(metrics["im7WorstDbc"])
```

### 8.1 通过主Analysis类分别读取IM3、IM5和IM7

当上层程序希望统一从 `Analysis` 导入测量接口时，可以直接传入PA输出和 `TwoToneWaveform`：

```python
from inc.lib.Analysis import Analysis

allMetrics = Analysis.AnalyzeTwoTone(
    paOutput,
    toneWaveform,
    parameters={"maximumOutputPowerDbm": 25.0, "width": 0},
)
im3Metrics = Analysis.CalculateIm3(paOutput, toneWaveform)
im5Metrics = Analysis.CalculateIm5(paOutput, toneWaveform)
im7Metrics = Analysis.CalculateIm7(paOutput, toneWaveform)

print(allMetrics["worstIntermodulationDbc"])
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
```

原始样值模式的规则如下：

- `sampleRateHz` 和 `toneFrequenciesHz` 是必填物理量；缺少任一项会报错，而不是猜测错误频率；
- `AnalyzeTwoTone` 对IM3、IM5和IM7均同时返回 `LowerDbc` 与 `UpperDbc`；每一阶的 `WorstDbc` 只是两侧中的较差者，不能替代两侧原始结果；
- 两个基波必须不同，且IM3、IM5和IM7理论位置都必须位于复Nyquist范围内；
- 原始发送数组和PA输出必须具有相同长度；
- 浮点样值默认 `width=0`；整数I/Q码必须在 `width` 参数或 `parameters["width"]` 中提供正确位宽；
- `TwoToneWaveform` 模式仍是最完整的调用方式。若同时提供它和 `sampleRateHz`、`toneFrequenciesHz` 或 `width`，这些值必须一致，避免把一个频率标签应用到另一段波形。

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
