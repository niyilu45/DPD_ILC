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

## 4. 有限记录RMS归一化

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

## 5. 浮点和定点公开接口

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

## 6. 采样率和防混叠

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

## 7. 参数列表

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

## 8. 最小使用示例

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

## 9. 适用范围和限制

1. 这是复基带双音，不直接生成实数射频载波。
2. 当前对象生成SISO向量；MIMO双音必须明确每链相位、功率和耦合场景后再扩展，不能简单复制。
3. 双音适合观察离散互调，不替代Wi-Fi调制EVM、频谱掩模或随机业务峰值测试。
4. 若仪表记录存在频偏，准确IM分析前应先用仪表本身或独立同步链校正频率轴。
5. PA热漂移会使迭代间互调缓慢变化；真实仪表测试应固定温度、平均次数和等待时间。
