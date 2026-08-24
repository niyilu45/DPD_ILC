# 最小系统隔离测试示例

## 1. 文档目标

本文件说明如何构造“只打开一个物理非理想”的最小系统。它适合回答以下问题：

- Tx I/Q不平衡本身会把IRR和EVM恶化到什么程度？
- FB I/Q不平衡是否只污染反馈观测，而不会改变forward结果？
- 固定结温从25摄氏度升到85摄氏度时，PA输出怎样变化？
- 相同PA输出功率下，热阻或占空比改变以后，温升和输出漂移怎样变化？

最小系统不是完整整机仿真。它的目的，是让输出变化可以归因到一个参数模块。确认单模块行为以后，才能把多个非理想逐个加入完整DPD系统。

## 2. 最小系统的共同规则

每个最小系统都遵守下面的隔离规则：

1. 每次只改变一个模块的参数组，其他非理想使用0或 `None`。
2. 所有对比使用同一个随机种子、同一段输入波形和同一参考面。
3. I/Q测试使用单位PA，排除PA压缩、记忆和温度影响。
4. Tx I/Q测试使用forward参考面，FB I/Q测试直接从已知PA输出进入 `ProcessPaOutput`。
5. 温漂测试只需重复调用 `Channel.Process(rawSignal, outputPowerDbm=...)`；函数内部暂停温度影响做参考校准，恢复原热状态后真实发射一次，不会闭环稳定当前热态输出。
6. 参数比较至少包含理想、轻微和压力三档，并给出预期单调趋势。
7. 理想双精度结果可能显示数百dB IRR或极小EVM，这只代表数值残差，不代表真实仪器动态范围。

最小系统的推荐执行顺序为：

```mermaid
flowchart LR
    input["固定输入与随机种子"] --> ideal["理想基线"]
    ideal --> one["只打开一个参数模块"]
    one --> sweep["轻微/典型/压力配置"]
    sweep --> metric["统一Analysis指标"]
    metric --> trend{"趋势符合物理预期？"}
    trend -->|否| reference["检查参考面、同步和归一化"]
    trend -->|是| next["再加入下一个模块"]
```

## 3. 最小系统A：单独测试Tx I/Q不平衡

### 3.1 系统边界

Tx I/Q误差位于PA之前。为了只观察I/Q镜像，本场景把PA替换为单位映射：

```math
x(n)
\longrightarrow
\alpha_{\mathrm{tx}}x(n)
+
\beta_{\mathrm{tx}}x^*(n)
\longrightarrow
y(n).
```

不添加PA非线性、耦合、噪声、CFO、SFO、FB I/Q或量化。输入使用proper complex随机信号，使 $x$ 和 $x^*$ 在有限样本上尽量独立，便于估计IRR。

### 3.2 参数对比

| 场景 | `txIqGainImbalanceDb` | `txIqPhaseImbalanceDegrees` | 目的 |
|---|---:|---:|---|
| Ideal | 0.0 | 0.0 | 数值基线 |
| Gain only | 0.3 | 0.0 | 单独观察增益误差 |
| Phase only | 0.0 | 2.0 | 单独观察正交误差 |
| Mild combined | 0.3 | 2.0 | 常用功能验证起点 |
| Stress combined | 1.0 | 5.0 | 明显镜像压力测试 |

### 3.3 最小可运行代码

```python
import numpy as np

from inc.lib.Analysis import Analysis
from inc.lib.Channel import Channel


class IdentityPa:
    """Provide a distortion-free PA boundary for impairment isolation."""

    def __init__(self) -> None:
        """Initialize one floating-point identity PA."""

        self.width = 0

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Return a defensive complex copy without electrical distortion."""

        return np.asarray(inputSignal, dtype=np.complex128).copy()

    def ProcessFloating(self, inputSignal: np.ndarray) -> np.ndarray:
        """Evaluate the same identity mapping at the floating boundary."""

        return self.Process(inputSignal)

    def SmallSignalGain(self) -> complex:
        """Return the exact direct small-signal gain."""

        return 1.0 + 0.0j


def GenerateProperComplexProbe(
    sampleCount: int,
    randomSeed: int,
) -> np.ndarray:
    """Generate one unit-RMS proper complex probe waveform."""

    randomGenerator = np.random.default_rng(randomSeed)
    probeSignal = (
        randomGenerator.standard_normal(sampleCount)
        + 1j * randomGenerator.standard_normal(sampleCount)
    ) / np.sqrt(2.0)
    return probeSignal / np.sqrt(np.mean(np.abs(probeSignal) ** 2))


def AnalyzeKnownWaveforms(
    referenceSignal: np.ndarray,
    measuredSignal: np.ndarray,
    sampleRateHz: float,
) -> dict:
    """Analyze a measured waveform without descriptor reconstruction."""

    resultAnalysis = Analysis(
        measuredSignal,
        transmittedSignal=referenceSignal,
        sampleRateHz=sampleRateHz,
        width=0,
    )
    return resultAnalysis.Analyze()


def RunTxIqComparison() -> dict:
    """Compare ideal, individual, and combined Tx I/Q impairments."""

    sampleRateHz = 80.0e6
    referenceSignal = GenerateProperComplexProbe(4096, 1001)
    scenarioDefinitions = {
        "Ideal": (0.0, 0.0),
        "Gain only": (0.3, 0.0),
        "Phase only": (0.0, 2.0),
        "Mild combined": (0.3, 2.0),
        "Stress combined": (1.0, 5.0),
    }
    comparisonResults = {}

    for scenarioName, (
        gainImbalanceDb,
        phaseImbalanceDegrees,
    ) in scenarioDefinitions.items():
        channel = Channel(
            paModel=IdentityPa(),
            parameters={
                "sampleMode": "forward",
                "sampleRateHz": sampleRateHz,
                "txIqGainImbalanceDb": gainImbalanceDb,
                "txIqPhaseImbalanceDegrees": phaseImbalanceDegrees,
                "txDcOffset": 0.0 + 0.0j,
                "noiseSnrDb": None,
                "width": 0,
            },
        )
        measuredSignal = channel.Process(referenceSignal)
        comparisonResults[scenarioName] = AnalyzeKnownWaveforms(
            referenceSignal,
            measuredSignal,
            sampleRateHz,
        )

    return comparisonResults


txIqResults = RunTxIqComparison()
for scenarioName, metrics in txIqResults.items():
    print(
        f"{scenarioName:16s} "
        f"IRR={metrics['irrDb']:8.2f} dB  "
        f"EVM={metrics['evmDb']:8.2f} dB"
    )
```

### 3.4 仿真结果与预期

上述确定性场景的典型结果为：

| 场景 | IRR | EVM | 预期判断 |
|---|---:|---:|---|
| Ideal | 约284 dB | 数值零残差 | 只表示双精度基线 |
| Gain only | 35.26 dB | -35.26 dB | 与0.3 dB增益误差公式一致 |
| Phase only | 35.16 dB | -35.16 dB | 与2度正交误差公式一致 |
| Mild combined | 32.20 dB | -32.20 dB | 两个镜像向量共同作用 |
| Stress combined | 22.83 dB | -22.83 dB | 镜像明显主导EVM |

这里EVM约等于IRR的负值，是因为系统中只有一个共轭镜像误差。若加入PA非线性、DC、噪声或削顶，该关系将不再严格成立。

### 3.5 隔离成功的判据

- 增益误差或相位误差增大时，IRR应降低、EVM应升高。
- `sampleMode="forward"` 仍然能看到Tx I/Q误差，因为它位于PA之前。
- `GetLastTransmitterOutput()` 应与单位PA输出一致。
- 如果Ideal场景仍只有二三十dB IRR，应先检查输入是否近似proper complex，以及Analysis是否使用了同一参考波形。

## 4. 最小系统B：单独测试FB I/Q不平衡

### 4.1 系统边界

本场景把输入定义为“已经存在的干净PA输出”，直接调用 `ProcessPaOutput`：

```math
y_{\mathrm{PA}}(n)
\longrightarrow
\alpha_{\mathrm{fb}}y_{\mathrm{PA}}(n)
+
\beta_{\mathrm{fb}}y_{\mathrm{PA}}^*(n)
\longrightarrow
z_{\mathrm{fb}}(n).
```

这样不会再次运行PA，也不会应用Tx I/Q。所有FB增益、FIR、时频偏、非线性、ADC和噪声都保持理想，只改变三个FB I/Q参数。

### 4.2 参数对比与代码

FB的增益和相位配置使用与Tx场景相同的五档值。把下面代码接在第3.3节公共函数之后即可运行：

```python
def RunFbIqComparison() -> dict:
    """Compare feedback I/Q settings on an already known PA output."""

    sampleRateHz = 80.0e6
    cleanPaOutput = GenerateProperComplexProbe(4096, 1001)
    scenarioDefinitions = {
        "Ideal": (0.0, 0.0),
        "Gain only": (0.3, 0.0),
        "Phase only": (0.0, 2.0),
        "Mild combined": (0.3, 2.0),
        "Stress combined": (1.0, 5.0),
    }
    comparisonResults = {}

    for scenarioName, (
        gainImbalanceDb,
        phaseImbalanceDegrees,
    ) in scenarioDefinitions.items():
        channel = Channel(
            parameters={
                "sampleMode": "fb",
                "sampleRateHz": sampleRateHz,
                "fbIqGainImbalanceDb": gainImbalanceDb,
                "fbIqPhaseImbalanceDegrees": phaseImbalanceDegrees,
                "fbDcOffset": 0.0 + 0.0j,
                "fbThirdOrderCoefficient": 0.0 + 0.0j,
                "fbClipAmplitude": None,
                "fbAdcWidth": None,
                "noiseSnrDb": None,
                "width": 0,
            }
        )
        measuredSignal = channel.ProcessPaOutput(cleanPaOutput)
        comparisonResults[scenarioName] = AnalyzeKnownWaveforms(
            cleanPaOutput,
            measuredSignal,
            sampleRateHz,
        )

    return comparisonResults


fbIqResults = RunFbIqComparison()
for scenarioName, metrics in fbIqResults.items():
    print(
        f"{scenarioName:16s} "
        f"IRR={metrics['irrDb']:8.2f} dB  "
        f"EVM={metrics['evmDb']:8.2f} dB"
    )
```

因为Tx与FB模块复用相同的I/Q系数换算，五档结果应与第3.4节基本一致。但是物理含义不同：

- Tx结果代表真实发射镜像，可以由增广DPD补偿。
- FB结果只是测量镜像，不应直接由发射DPD补偿。
- 把同一组非零 `fbIq...` 参数放入 `sampleMode="forward"` 后，输出应回到Ideal基线。这是区分Tx和FB参考面的关键对照。

### 4.3 单独测试DC偏置

IRR描述共轭镜像，不适合度量纯DC。测试 `fbDcOffset` 时应主要观察EVM和零频谱线：

```python
def RunFbDcComparison() -> dict:
    """Compare feedback DC magnitude without adding image imbalance."""

    sampleRateHz = 80.0e6
    cleanPaOutput = GenerateProperComplexProbe(4096, 1001)
    dcMagnitudes = (0.0, 0.001, 0.01, 0.03)
    comparisonResults = {}

    for dcMagnitude in dcMagnitudes:
        channel = Channel(
            parameters={
                "sampleMode": "fb",
                "fbDcOffset": dcMagnitude + 0.0j,
                "width": 0,
            }
        )
        measuredSignal = channel.ProcessPaOutput(cleanPaOutput)
        comparisonResults[dcMagnitude] = AnalyzeKnownWaveforms(
            cleanPaOutput,
            measuredSignal,
            sampleRateHz,
        )

    return comparisonResults
```

| `|fbDcOffset|` | 约占单位RMS信号 | EVM典型结果 |
|---:|---:|---:|
| 0 | 0% | 数值零残差 |
| 0.001 | 0.1% | 约 -60.01 dB |
| 0.01 | 1% | 约 -40.01 dB |
| 0.03 | 3% | 约 -30.46 dB |

DC增大时EVM按约20对数规律变差，但IRR仍可能很高，因为DC不是 $x^*$ 镜像。只看IRR会漏掉这一类误差。

## 5. 最小系统C：单独测试固定温度角

### 5.1 测试目的

固定温度角不模拟自热过程，只回答“同一参考温度校准驱动在不同结温下输出怎样变化”。它最适合隔离温度到电参数的映射；示例由 `Channel.Process` 在内部完成无热校准和恢复后真实处理：

```math
T_j
\longrightarrow
G(T_j),\ \phi(T_j),\ A_{\mathrm{sat}}(T_j),\ \eta_{\mathrm{nl}}(T_j)
\longrightarrow
y(n,T_j).
```

本例使用25、55和85摄氏度三点。温度系数故意设置得较明显，用于确认趋势，不代表具体器件规格。

### 5.2 三种热模型的推荐起始配置

`ThermalConfig.Recommended` 会同时设置全部21个热参数并执行合法性校验。下面三套配置都能直接送入 `PaModel`；`sampleRateHz` 必须替换成实际波形采样率：

```python
from inc.lib.PaModel import ThermalConfig


staticThermalConfig = ThermalConfig.Recommended(
    "static",
    sampleRateHz=80.0e6,
)
singleRcThermalConfig = ThermalConfig.Recommended(
    "single_rc",
    sampleRateHz=80.0e6,
)
fosterThermalConfig = ThermalConfig.Recommended(
    "foster",
    sampleRateHz=80.0e6,
)

print(staticThermalConfig)
print(singleRcThermalConfig)
print(fosterThermalConfig)
```

关键动态参数分别为：

| 模型 | 推荐热阻 | 推荐时间常数 | 预期行为 |
|---|---|---|---|
| `static` | `(1.0,)`占位 | `(1.0,)`占位 | 固定在55摄氏度；推荐另扫25/55/85摄氏度 |
| `single_rc` | `(20.0,)` 摄氏度/W | `(20e-3,)` s | 一条平滑指数升温和冷却曲线 |
| `foster` | `(2.0,8.0,20.0)` 摄氏度/W | `(50e-6,5e-3,0.5)` s | 同时包含快、中、慢热响应 |

以下是最小Foster运行例。用户只调用Channel主入口，不需要关闭温度模型或显式执行功率校准：

```python
from inc.lib.Channel import Channel
from inc.lib.PaModel import PaModel, ThermalConfig


fosterThermalConfig = ThermalConfig.Recommended(
    "foster",
    sampleRateHz=80.0e6,
)
fosterPa = PaModel(
    parameters={
        "modelName": "gmp",
        "thermalConfig": fosterThermalConfig,
        "width": 0,
    }
)
fosterChannel = Channel(
    paModel=fosterPa,
    parameters={
        "sampleRateHz": 80.0e6,
        "maximumOutputPowerDbm": 25.0,
        "width": 0,
    },
)

for frameIndex in range(20):
    receivedSignal = fosterChannel.Process(
        rawSignal,
        outputPowerDbm=20.0,
    )
    print(frameIndex, fosterChannel.GetThermalMetrics())
    fosterChannel.AdvanceThermalIdle(1.0e-3)
```

全部公共参数、四个温度电系数、敏感性范围和实测替换规则见 [PaModel.md的完整推荐表](./PaModel.md#1371-三种已实现模型的完整推荐值)。

### 5.3 压力对比使用的公共温度测试工具

下面的固定温度和动态热阻场景故意放大温度电参数，以便自动测试在短记录内稳定观察到趋势。这组放大值不能代替上面的工程推荐起点。

```python
import numpy as np

from inc.lib.Analysis import Analysis
from inc.lib.Channel import Channel
from inc.lib.PaModel import PaModel, ThermalConfig


def GenerateThermalProbe(
    sampleCount: int,
    randomSeed: int,
) -> np.ndarray:
    """Generate one amplitude-varying unit-RMS thermal probe."""

    randomGenerator = np.random.default_rng(randomSeed)
    probeSignal = (
        randomGenerator.standard_normal(sampleCount)
        + 1j * randomGenerator.standard_normal(sampleCount)
    ) / np.sqrt(2.0)
    return probeSignal / np.sqrt(np.mean(np.abs(probeSignal) ** 2))


def CreateThermalChannel(
    sampleRateHz: float,
    modelName: str,
    initialTemperatureC: float,
    thermalResistanceCPerW: float,
    thermalTimeConstantSec: float,
) -> Channel:
    """Create one isolated SISO thermal PA and its forward channel."""

    thermalConfig = ThermalConfig(
        enabled=True,
        modelName=modelName,
        sampleRateHz=sampleRateHz,
        ambientTemperatureC=25.0,
        initialJunctionTemperatureC=initialTemperatureC,
        referenceTemperatureC=25.0,
        thermalResistancesCPerW=(thermalResistanceCPerW,),
        thermalTimeConstantsSec=(thermalTimeConstantSec,),
        thermalUpdateIntervalSamples=100,
        idleDissipatedPowerW=0.0,
        referenceOutputPowerDbm=25.0,
        gainTemperatureCoefficientDbPerC=-0.08,
        phaseTemperatureCoefficientDegreesPerC=0.10,
        saturationTemperatureCoefficientPerC=-0.004,
        nonlinearityTemperatureCoefficientPerC=0.010,
        maximumJunctionTemperatureC=150.0,
    )
    paModel = PaModel(
        parameters={
            "modelName": "wiener",
            "thermalConfig": thermalConfig,
            "width": 0,
        }
    )
    return Channel(
        paModel=paModel,
        parameters={
            "sampleRateHz": sampleRateHz,
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.05,
            "width": 0,
        },
    )
```

### 5.4 固定温度角对比代码

```python
def RunStaticTemperatureComparison() -> dict:
    """Compare one hidden reference calibration at fixed temperatures."""

    sampleRateHz = 100.0e3
    rawSignal = GenerateThermalProbe(2000, 2027)
    temperatureValuesC = (25.0, 55.0, 85.0)
    outputSignals = {}
    thermalMetrics = {}

    for temperatureC in temperatureValuesC:
        channel = CreateThermalChannel(
            sampleRateHz=sampleRateHz,
            modelName="static",
            initialTemperatureC=temperatureC,
            thermalResistanceCPerW=20.0,
            thermalTimeConstantSec=0.02,
        )
        outputSignals[temperatureC] = channel.Process(
            rawSignal,
            outputPowerDbm=20.0,
        )
        thermalMetrics[temperatureC] = channel.GetThermalMetrics()

    coldReference = outputSignals[25.0]
    comparisonResults = {}
    for temperatureC in temperatureValuesC:
        driftMetrics = Analysis(
            outputSignals[temperatureC],
            transmittedSignal=coldReference,
            sampleRateHz=sampleRateHz,
            width=0,
        ).Analyze()
        comparisonResults[temperatureC] = {
            "outputPowerDbm": thermalMetrics[temperatureC][
                "outputPowerDbm"
            ],
            "evmDbVersusCold": driftMetrics["evmDb"],
            "evmPercentVersusCold": driftMetrics["evmPercent"],
        }

    return comparisonResults


staticTemperatureResults = RunStaticTemperatureComparison()
for temperatureC, metrics in staticTemperatureResults.items():
    print(temperatureC, metrics)
```

### 5.4 固定温度角结果

| 结温 | 输出功率 | 相对25摄氏度EVM | EVM百分比 |
|---:|---:|---:|---:|
| 25摄氏度 | 19.97 dBm | 数值零残差 | 0% |
| 55摄氏度 | 16.07 dBm | -21.64 dB | 8.28% |
| 85摄氏度 | 11.82 dBm | -14.92 dB | 17.95% |

这个结果由示例中较强的负增益温度系数、负饱和温度系数和正附加非线性系数共同产生。若只打开 `gainTemperatureCoefficientDbPerC`，公共复增益对齐会消除大部分EVM变化，但绝对输出功率仍会漂移；因此温漂测试必须同时观察输出功率和对齐后EVM。

## 6. 最小系统D：单独测试动态自热

### 6.1 测试边界

动态自热测试仍包含“无热参考校准”和“真实温度发射”两个物理阶段，但它们已经封装在一次 `Channel.Process(rawSignal, outputPowerDbm=...)` 调用内部。用户不用关闭温度模型、调用校准器或管理 `frozenInput`。

每帧传入的 `outputPowerDbm` 只约束暂停温度影响后的参考电模型。Channel随后恢复本帧开始前的结温和累计时间，并用收敛输入真实发射一次；它不会围绕当前热态输出重新稳定功率。因此连续调用仍会得到自然温升和输出漂移。

### 6.2 比较不同热阻

保持时间常数20 ms、输入波形、目标功率和温度系数不变，只改变单RC热阻：5、20和40摄氏度每瓦。热阻越大，相同耗散功率形成的稳态温升越高。

```python
def RunThermalResistanceScenario(
    thermalResistanceCPerW: float,
) -> dict:
    """Measure eight internally calibrated thermal frames."""

    sampleRateHz = 100.0e3
    rawSignal = GenerateThermalProbe(2000, 2027)
    channel = CreateThermalChannel(
        sampleRateHz=sampleRateHz,
        modelName="single_rc",
        initialTemperatureC=25.0,
        thermalResistanceCPerW=thermalResistanceCPerW,
        thermalTimeConstantSec=0.02,
    )
    firstOutput = channel.Process(
        rawSignal,
        outputPowerDbm=20.0,
    )
    firstMetrics = channel.GetThermalMetrics()
    lastOutput = firstOutput
    lastMetrics = firstMetrics
    for _ in range(7):
        lastOutput = channel.Process(
            rawSignal,
            outputPowerDbm=20.0,
        )
        lastMetrics = channel.GetThermalMetrics()

    driftMetrics = Analysis(
        lastOutput,
        transmittedSignal=firstOutput,
        sampleRateHz=sampleRateHz,
        width=0,
    ).Analyze()
    return {
        "firstTemperatureC": firstMetrics[
            "endingJunctionTemperatureC"
        ],
        "lastTemperatureC": lastMetrics[
            "endingJunctionTemperatureC"
        ],
        "firstOutputPowerDbm": firstMetrics["outputPowerDbm"],
        "lastOutputPowerDbm": lastMetrics["outputPowerDbm"],
        "evmDbVersusFirstFrame": driftMetrics["evmDb"],
        "evmPercentVersusFirstFrame": driftMetrics["evmPercent"],
    }


for thermalResistanceCPerW in (5.0, 20.0, 40.0):
    print(
        thermalResistanceCPerW,
        RunThermalResistanceScenario(thermalResistanceCPerW),
    )
```

典型结果为：

| 热阻 | 第1帧结温 | 第8帧结温 | 第1帧功率 | 第8帧功率 | 第8帧相对第1帧EVM |
|---:|---:|---:|---:|---:|---:|
| 5摄氏度每瓦 | 25.52摄氏度 | 25.82摄氏度 | 19.94 dBm | 19.87 dBm | -51.14 dB |
| 20摄氏度每瓦 | 27.05摄氏度 | 28.13摄氏度 | 19.83 dBm | 19.58 dBm | -39.61 dB |
| 40摄氏度每瓦 | 29.00摄氏度 | 30.89摄氏度 | 19.69 dBm | 19.23 dBm | -33.89 dB |

热阻增大后，温升、功率下降和波形形状漂移都增大，符合单RC模型预期。若结温升高但输出功率和EVM完全不变，应检查所有温度电参数是否仍为0。

### 6.3 比较不同占空比

占空比测试保持活动区目标输出功率相同，只改变每帧中有效RF样点的比例。补零样点不进入活动区输出功率统计，但仍推进热时间并按 `idleDissipatedPowerW` 处理。

```python
def RunDutyCycleScenario(activeDutyCycle: float) -> dict:
    """Measure eight equal-duration frames at one RF duty cycle."""

    sampleRateHz = 100.0e3
    fullProbe = GenerateThermalProbe(2000, 2027)
    activeSampleCount = int(fullProbe.size * activeDutyCycle)
    rawSignal = np.zeros(fullProbe.size, dtype=np.complex128)
    rawSignal[:activeSampleCount] = fullProbe[:activeSampleCount]
    channel = CreateThermalChannel(
        sampleRateHz=sampleRateHz,
        modelName="single_rc",
        initialTemperatureC=25.0,
        thermalResistanceCPerW=20.0,
        thermalTimeConstantSec=0.02,
    )
    for _ in range(8):
        channel.Process(
            rawSignal,
            outputPowerDbm=20.0,
        )

    thermalMetrics = channel.GetThermalMetrics()
    return {
        "measuredDutyCycle": thermalMetrics["activeSampleDutyCycle"],
        "endingTemperatureC": thermalMetrics[
            "endingJunctionTemperatureC"
        ],
        "outputPowerDbm": thermalMetrics["outputPowerDbm"],
        "averageDissipatedPowerW": thermalMetrics[
            "averageDissipatedPowerW"
        ],
    }


for activeDutyCycle in (0.25, 0.50, 1.00):
    print(activeDutyCycle, RunDutyCycleScenario(activeDutyCycle))
```

典型结果为：

| 配置占空比 | 测得占空比 | 第8帧结温 | 活动区输出功率 | 平均耗散功率 |
|---:|---:|---:|---:|---:|
| 25% | 25.00% | 25.55摄氏度 | 19.89 dBm | 0.041 W |
| 50% | 49.95% | 26.22摄氏度 | 19.78 dBm | 0.081 W |
| 100% | 99.95% | 28.13摄氏度 | 19.58 dBm | 0.157 W |

占空比越高，平均耗散越大、结温越高，同一参考温度目标下的真实输出功率下降越明显。示例每帧都传入20 dBm，但Channel只在暂停温度影响的参考电模型上校准，不会闭环追踪当前热态输出，因此不会隐藏自然温漂。只有另行构造一个根据热态实测功率持续修正驱动的外部控制环，才会把这种功率差异抵消。

### 6.4 从实际PA测量得到上述参数

本章示例用于隔离参数影响，不应从仿真曲线反向猜测真实器件参数。实际工程应同步测量结温、DC电压电流、输入/输出RF功率和I/Q，先拟合热源效率，再拟合单RC或Foster，最后在已知结温下拟合增益、相位、饱和尺度和附加非线性温度系数。完整仪表连接、测试时序、NumPy拟合函数、static/单RC/Foster回填例和留出数据验收方法见 [PA温度特性测量、模型辨识与参数回填](./PaThermalMeasurement.md)。

## 7. 参数对比时应该记录哪些结果

### 7.1 I/Q最小系统

至少记录：

- `irrDb`：判断共轭镜像强度；
- `evmDb` 和 `evmPercent`：判断镜像或DC对调制误差的贡献；
- Tx测试的 `GetLastTransmitterOutput()`：确认误差位于PA前；
- forward与fb成对结果：区分真实发射误差和反馈接收机误差。

### 7.2 温漂最小系统

至少记录：

- `startingJunctionTemperatureC` 和 `endingJunctionTemperatureC`；
- `averageDissipatedPowerW` 和 `activeSampleDutyCycle`；
- `outputPowerDbm`：观察绝对增益和压缩漂移；
- 相对冷机或第1帧的EVM：观察公共复增益对齐后剩余的非线性形状变化；
- ACLR或双音IM3：需要判断温度相关带外再生时再加入，不能只看功率。

## 8. 如何扩展到其他最小系统

同一方法可以扩展到其他Channel模块。保持所有未测试模块理想，只改变表中的目标参数：

| 最小系统 | 只改变的参数 | 推荐三档对比 | 主要指标 |
|---|---|---|---|
| PA前耦合 | `prePaCouplingPaths` | `None`、-30 dB、-20 dB | 非对角泄漏、EVM、联合校准收敛 |
| PA后耦合 | `postPaCouplingPaths` | `None`、-30 dB、-20 dB | 接收端泄漏、EVM、去嵌入残差 |
| FB同步 | Delay、CFO、SFO | 0、典型值、压力值 | 估计误差、EVM、帧尾残差 |
| FB非线性 | `fbThirdOrderCoefficient` | 0、0.01、0.10幅度 | EVM、ACLR、双音IM3 |
| FB ADC | `fbAdcWidth` | 16、12、8 bit | 量化EVM、削顶率、噪声底 |
| 白噪声 | `noiseSnrDb` | 45、35、25 dB | SNR和EVM地板 |

扩展时不要直接复制完整整机配置。最小系统的价值正是把参考面、输入、随机种子和其他模块固定下来，只让一个配置组产生变化。

## 9. 结果不符合预期时的检查顺序

1. 确认比较使用同一输入数组，而不是相同配置重新生成的另一段随机波形。
2. 确认 `width=0`，先排除公开定点量化；验证通过后再单独加入定点系统。
3. 确认Analysis使用 `transmittedSignal=referenceSignal`，没有进入盲Descriptor解析。
4. Tx I/Q必须通过 `Process`，FB I/Q必须通过fb模式或 `ProcessPaOutput`；不要混淆参考面。
5. 温漂阶段可以继续传入 `outputPowerDbm`；当前Channel只在暂停温度影响的参考面校准，恢复原热状态后真实处理一次，不会闭环稳定当前热态输出。
6. 检查温度系数是否非零；只有热网络升温而电参数不随温度变化时，波形可能几乎不变。
7. 一次只恢复一个被关闭的模块，直到找到趋势改变的来源。

## 10. 最小系统E：单独验证Rapp无记忆PA

### 10.1 为什么需要这个场景

Rapp是本工程新增的严格无记忆固态PA模型。它只让当前输出依赖当前输入：

```math
y[n]
=
\frac{G x[n]}
{\left(1+\left(\frac{|x[n]|}{A_{\mathrm{sat}}}\right)^{2p}\right)^{1/(2p)}}.
```

因此，只要两段波形在第 $n$ 点的复数值相同，不论此前样点如何变化，Rapp都必须给出完全相同的 $y[n]$。这个场景的目的不是证明Rapp“没有非线性”，而是把**静态压缩**和**动态记忆**拆开：Rapp可以产生明显IM3、IM5和IM7，但不应产生真正的频率响应起伏、群时延或动态迟滞。

### 10.2 三档膝点平滑度对比

固定 `linearGain=1.0`、`saturationAmplitude=1.0`，只改变 `rappSmoothness`：

| `rappSmoothness` | 适合的对照目的 | $|x|=0.5$增益压缩 | $|x|=1.0$增益压缩 | $|x|=1.4$增益压缩 |
|---:|---|---:|---:|---:|
| 1.5 | 很软的膝点，较早出现渐进压缩 | -0.341 dB | -2.007 dB | -3.822 dB |
| 3.0 | 推荐默认值，代表常见平滑SSPA | -0.022 dB | -1.003 dB | -3.103 dB |
| 8.0 | 接近硬限幅，用于压力测试 | 约0 dB | -0.376 dB | -2.925 dB |

这些数值是相对于小信号线性增益 $G$ 的压缩量。`rappSmoothness` 越大，膝点前越接近直线、膝点附近转折越突然；它不会引入记忆或AM-PM。

### 10.3 最小可运行代码：相同当前样点、不同历史

```python
import numpy as np

from inc.lib.PaModel import PaModel, RappConfig, WienerConfig


sampleCount = 256
comparisonIndex = 128
firstHistory = np.zeros(sampleCount, dtype=np.complex128)
secondHistory = np.zeros(sampleCount, dtype=np.complex128)

# Only the preceding sample is different.  The compared current sample is exact.
firstHistory[comparisonIndex - 1] = 0.9 + 0.2j
secondHistory[comparisonIndex - 1] = -0.3 + 0.7j
firstHistory[comparisonIndex] = 0.6 - 0.1j
secondHistory[comparisonIndex] = firstHistory[comparisonIndex]

rappPa = PaModel(
    modelName="rapp",
    rappConfig=RappConfig(
        linearGain=1.0,
        saturationAmplitude=1.0,
        rappSmoothness=3.0,
    ),
    width=0,
)
wienerPa = PaModel(
    modelName="wiener",
    wienerConfig=WienerConfig(linearTaps=(0.90 + 0.0j, 0.10 + 0.0j)),
    width=0,
)

rappFirst = rappPa.Process(firstHistory)
rappSecond = rappPa.Process(secondHistory)
wienerFirst = wienerPa.Process(firstHistory)
wienerSecond = wienerPa.Process(secondHistory)

print("Rapp context difference:", abs(rappFirst[comparisonIndex] - rappSecond[comparisonIndex]))
print("Wiener context difference:", abs(wienerFirst[comparisonIndex] - wienerSecond[comparisonIndex]))
```

预期结果：Rapp差值处于浮点舍入误差量级；Wiener差值明显非零，因为其第二个 `linearTaps` 抽头显式使用 $x[n-1]$。进一步运行 `RunPaCharacterizationBenchmark` 时，还应看到Rapp的频响起伏、群时延和动态迟滞接近零，而其互调随输出功率升高而恶化。这样才能确认测试链既能发现静态非线性，也不会把无记忆模型误判成有记忆。
