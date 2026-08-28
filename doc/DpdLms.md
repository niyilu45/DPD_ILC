# `DpdLms.py`程序使用手册

`inc/lib/DpdLms.py`提供基于GMP特征的复数LMS和NLMS数字预失真器。它继承 `DpdGmp` 的基函数顺序、批量推理、输出限幅和定点边界，同时增加：

- 每到一个样点就更新一次影子系数；
- 帧冻结或运行特征尺度；
- 样点提交或帧提交；
- 恒等先验泄漏；
- 样点权重上限；
- 单次系数步进范数限制；
- 支持原始长度不同的PA输入和反馈输出间接学习；
- 更新前、在线和更新后三种NMSE诊断。

详细数学推导见 [DPD-LMS逐样点原理](./DPD-LMS.md)。

---

## 1. 类与返回对象

### 1.1 `DpdLms`

构造函数为：

```python
DpdLms(
    parameters=None,
    width=None,
    **parameterOverrides,
)
```

默认值在构造函数内部建立，再通过 `ChainMap` 与调用方覆盖合并：

```text
显式关键字覆盖
    >
parameters外部活动映射
    >
类内部不可变默认值
```

调用方不需要创建 `ChainMap`。未知键产生 `UserWarning`、被忽略，其余已识别参数继续生效。

### 1.2 `DpdLmsTrainingResult`

`UpdateFromLabels` 和 `UpdateIndirect` 返回不可变结果：

| 字段 | 含义 |
|---|---|
| `sampleCount` | 进入本次有序训练的样点数 |
| `updateCount` | 实际执行系数更新的次数 |
| `featureCount` | 当前GMP复系数数目 |
| `adaptationMode` | `"lms"`或`"nlms"` |
| `beforeNmseDb` | 训练前固定影子系数的整帧NMSE |
| `onlineNmseDb` | 系数随样点变化时累计的训练NMSE |
| `afterNmseDb` | 最终固定影子系数重新处理整帧的NMSE |
| `coefficientUpdateNorm` | 本帧开始到结束的系数变化范数 |
| `maximumSampleUpdateNorm` | 帧内最大的单样点系数变化范数 |
| `coefficientsCommitted` | 完成接口是否已把影子系数提交给活动模型 |

`ToDict()`返回普通字典。

---

## 2. 完整参数表

`DpdLms`保留全部 `DpdGmp` 参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `nonlinearOrders` | `(1, 3, 5, 7)` | GMP奇数阶集合，必须包含1 |
| `memoryDepth` | `3` | main支路复载波记忆深度 |
| `crossMemoryDepth` | `2` | lagging/leading包络交叉深度 |
| `ridgeFactor` | `1e-6` | 仅供继承的批量训练方法使用；LMS更新不构造岭矩阵 |
| `coefficientLearningRate` | `1.0` | 仅供继承的批量训练方法使用，不是逐样点步长 |
| `chunkSize` | `8192` | 帧尺度统计和固定模型评估的分块长度 |
| `peakWeightExponent` | `0.0` | `UpdateFromLabels`自动生成的包络峰值权重 |
| `maximumOutputMagnitude` | `2.0` | 部署 `Process` 的输出包络上限 |
| `width` | `16` | 0为浮点；正数为公开I/Q整数码位宽 |

新增逐样点参数：

| 参数 | 默认值 | 约束 | 作用 |
|---|---:|---|---|
| `adaptationMode` | `"nlms"` | `"lms"`或`"nlms"` | 选择普通LMS或归一化LMS |
| `learningRate` | `0.05` | 正有限数 | 每个有效更新样点的步长 |
| `normalizationEpsilon` | `1e-6` | 正有限数 | 特征尺度和NLMS分母保护 |
| `leakageFactor` | `1e-7` | `[0,1)` | 把长期漂移缓慢拉回恒等DPD |
| `featureScaleMode` | `"frame"` | `"frame"`或`"running"` | 完整帧冻结尺度或指数运行尺度 |
| `featurePowerForgettingFactor` | `0.999` | `[0,1)` | 运行特征功率的遗忘因子 |
| `updateDecimation` | `1` | 正整数 | 每隔多少个样点更新一次；1为严格逐样点 |
| `coefficientCommitMode` | `"frame"` | `"frame"`或`"sample"` | 活动系数在帧末或每个样点后生效 |
| `maximumSampleUpdateNorm` | `0.05` | 正数或`None` | 单样点系数步进范数上限 |
| `maximumSampleWeight` | `8.0` | 正有限数 | 峰值或外部样点权重上限 |

特别注意：

- `learningRate`才是LMS逐样点步长；
- `coefficientLearningRate`属于父类批量岭回归；
- 两者名称不同，不能混用。

---

## 3. 方法总览

| 方法 | 输入 | 输出 | 用途 |
|---|---|---|---|
| `GetParameters()` | 无 | `dict` | 获取当前ChainMap解析结果 |
| `UpdateParameters(**overrides)` | 参数覆盖 | `None` | 更新参数；结构改变时恢复恒等状态 |
| `GetFeatureSpecs()` | 无 | `tuple` | 查看系数与GMP特征的固定对应关系 |
| `GetCoefficients()` | 无 | 数组 | 获取活动部署系数 |
| `GetAdaptiveCoefficients()` | 无 | 数组 | 获取逐样点更新的影子系数 |
| `SetCoefficients(coefficients)` | 复系数数组 | `None` | 同时设置活动和影子系数 |
| `ResetCoefficients()` | 无 | `None` | 活动和影子系数恢复恒等DPD |
| `ResetAdaptiveState(copyActiveCoefficients=True)` | 是否复制活动系数 | `None` | 清除历史、尺度和统计 |
| `PrepareFeatureScale(referenceSignal)` | 完整公开参考帧 | 尺度数组 | 只计算每个GMP特征的帧RMS |
| `BeginFrame(referenceSignal=None)` | 帧尺度模式需要完整参考 | 尺度数组 | 清除历史并开始独立帧 |
| `BuildFeatureVector(referenceSample)` | 一个内部浮点样点 | 一行GMP特征 | 硬件移植时的特征核参考 |
| `UpdateSampleFloating(referenceSample, targetSample, sampleWeight=1)` | 内部归一化样点 | 更新前预测 | 内部浮点逐样点核 |
| `UpdateSample(referenceSample, targetSample, sampleWeight=1)` | 公开浮点或定点样点 | 同公开格式预测 | 对外逐样点入口 |
| `CommitCoefficients()` | 无 | `None` | 一次性把影子系数复制为活动系数 |
| `UpdateFromLabels(referenceSignal, targetSignal, sampleWeights=None)` | 完整有序标签对 | `DpdLmsTrainingResult` | 软件中按时间顺序逐点回放 |
| `UpdateIndirect(paInputSignal, paOutputSignal, sampleRateHz, signalProcessingParameters=None, sampleWeights=None, paOutputFullScaleAmplitude=1.0)` | PA输入和任意长度反馈采集 | `DpdLmsTrainingResult` | 同步后逐点训练后置逆；末尾参数声明反馈输出码轨的物理量程 |
| `Process(inputSignal)` | 新DPD参考波形 | 预失真波形 | 使用活动系数部署 |
| `CalculateNmse(referenceSignal, targetSignal, sampleWeights=None)` | 标签对 | dB标量 | 评估活动固定模型 |
| `GetLastLmsTrainingResult()` | 无 | 结果或`None` | 获取最近一次完整逐点训练摘要 |

---

## 4. 最小逐样点程序

工程根目录提供可直接运行的：

```text
SmallestLMS.py
```

运行：

```powershell
python SmallestLMS.py
```

该程序故意不调用Wi-Fi、PA、ILC、Analysis或批量训练接口，只保留移植所需的逐样点骨架：

```python
import numpy as np

from inc.lib.DpdLms import DpdLms


randomGenerator = np.random.default_rng(7)
referenceSignal = (
    randomGenerator.standard_normal(8192)
    + 1j * randomGenerator.standard_normal(8192)
)
referenceSignal *= 0.25 / np.sqrt(
    np.mean(np.abs(referenceSignal) ** 2)
)

targetSignal = (
    1.03 * referenceSignal
    + 0.18
    * referenceSignal
    * np.abs(referenceSignal) ** 2
)

dpdLms = DpdLms(
    parameters={
        "nonlinearOrders": (1, 3),
        "memoryDepth": 1,
        "crossMemoryDepth": 0,
        "adaptationMode": "nlms",
        "learningRate": 0.10,
        "featureScaleMode": "frame",
        "coefficientCommitMode": "frame",
        "maximumOutputMagnitude": None,
        "width": 0,
    }
)

# This is the only complete-frame normalization pass.
dpdLms.BeginFrame(referenceSignal)

# This loop maps directly to one software, DSP, or HDL sample clock.
for referenceSample, targetSample in zip(
    referenceSignal,
    targetSignal,
):
    dpdLms.UpdateSample(
        complex(referenceSample),
        complex(targetSample),
    )

# Shadow coefficients changed every sample; deployment changes once here.
dpdLms.CommitCoefficients()

print(dpdLms.GetFeatureSpecs())
print(dpdLms.GetCoefficients())
```

期望恢复：

```text
('main', 1, 0, 0) approximately 1.03
('main', 3, 0, 0) approximately 0.18
```

### 4.1 移植时必须保持的操作顺序

每个样点的硬件或C实现必须保持：

```text
push reference sample
    ->
build feature vector
    ->
predict with old shadow coefficients
    ->
calculate target minus prediction
    ->
calculate normalized update
    ->
limit update norm
    ->
write all new shadow coefficients
```

不能先更新部分系数，再用新旧混合系数完成同一样点预测。

### 4.2 与最小程序不同的真实反馈输入

最小程序的 `targetSignal` 是已知数学标签。在真实间接学习中：

- `referenceSample`来自同步后的PA反馈输出；
- `targetSample`来自当时实际送入PA的数字样点；
- 每个反馈样点必须和对应PA输入样点处在相同时间索引。

---

## 5. 完整数组接口仍然是逐样点更新

如果已有完整标签数组，可以调用：

```python
from inc.lib.DpdLms import DpdLms


dpdLms = DpdLms(
    parameters={
        "nonlinearOrders": (1, 3, 5),
        "memoryDepth": 3,
        "crossMemoryDepth": 1,
        "learningRate": 0.05,
        "featureScaleMode": "frame",
        "coefficientCommitMode": "frame",
        "width": 0,
    }
)

trainingResult = dpdLms.UpdateFromLabels(
    referenceSignal,
    targetSignal,
)

predistortedSignal = dpdLms.Process(referenceSignal)
print(trainingResult.ToDict())
```

`UpdateFromLabels`接收一批数据，不代表它执行批量最小二乘。内部仍然是：

```python
for sampleIndex in range(referenceSignal.size):
    UpdateSampleFloating(...)
```

完整数组只用于：

1. 公开定点边界统一解码；
2. 帧尺度预统计；
3. 更新前固定NMSE；
4. 严格按时间顺序逐样点更新；
5. 更新后固定NMSE；
6. 返回完整训练摘要。

它不会构造 $K\times K$ 正规矩阵，也不会调用 `numpy.linalg.solve`。

---

## 6. 逐样点与一批数据处理的程序差异

### 6.1 `DpdGmp.UpdateCoefficients`

批量DpdGmp分为三次数据遍历：

1. 统计每个特征列RMS；
2. 累积正规矩阵和目标投影；
3. 求解一次系数并重新评估更新前后NMSE。

核心状态为：

```text
normalMatrix[K, K]
targetProjection[K]
featureScale[K]
```

系数只在整批结束时变化一次。

### 6.2 `DpdLms.UpdateFromLabels`

逐样点DpdLms执行：

1. 可选的一次帧尺度统计；
2. 初始化因果历史；
3. 按时间顺序访问每个样点；
4. 每个样点构造一个 `featureVector[K]`；
5. 每个有效样点更新一次 `adaptiveCoefficients[K]`；
6. 帧末提交；
7. 使用最终固定系数评估。

核心状态为：

```text
sampleHistory[M + L]
featureScale[K]
featurePower[K]
adaptiveCoefficients[K]
activeCoefficients[K]
```

没有 `normalMatrix[K, K]`。

### 6.3 为逐样点模式专门增加的处理

| 特殊处理 | 对应代码 | 原因 |
|---|---|---|
| 因果历史缓存 | `BuildFeatureVector` | 当前样点调用没有完整过去数组切片 |
| 帧/运行尺度 | `PrepareFeatureScale`、`ResolveFeatureScale` | 保持高阶特征数值尺度 |
| 影子系数 | `adaptiveCoefficients` | 避免训练中的部分系数暴露给部署 |
| 活动系数 | 继承的 `coefficients` | `Process`始终使用一个完整稳定向量 |
| 帧末原子提交 | `CommitCoefficients` | 避免OFDM帧内时变系数产生额外频谱 |
| 恒等泄漏 | `leakageFactor` | 比把DPD拉向全零更符合安全初值 |
| 权重上限 | `maximumSampleWeight` | 防止单个峰值破坏稳定性 |
| 单步投影 | `maximumSampleUpdateNorm` | 阻止反馈毛刺造成巨大系数跳变 |
| 三种NMSE | `DpdLmsTrainingResult` | 区分固定前、时变在线、固定后三种模型 |
| 更新抽取 | `updateDecimation` | 支持硬件降低自适应速率 |

---

## 7. 间接学习示例

PA输入和反馈输出可以长度不同：

```python
from inc.lib.DpdLms import DpdLms


dpdLms = DpdLms(
    parameters={
        "nonlinearOrders": (1, 3, 5, 7),
        "memoryDepth": 3,
        "crossMemoryDepth": 2,
        "adaptationMode": "nlms",
        "learningRate": 0.05,
        "coefficientCommitMode": "frame",
        "width": 0,
    }
)

trainingResult = dpdLms.UpdateIndirect(
    paInputSignal=actualPaInput,
    paOutputSignal=feedbackCapture,
    sampleRateHz=80.0e6,
    signalProcessingParameters={
        "enableIntegerDelayCompensation": True,
        "enableFractionalDelayCompensation": True,
        "enableCarrierFrequencyOffsetCompensation": True,
        "enableSamplingFrequencyOffsetCompensation": True,
        "enableComplexGainCompensation": True,
    },
    paOutputFullScaleAmplitude=2.0,
)

nextPaInput = dpdLms.Process(nextReferenceSignal)
print(trainingResult.ToDict())
```

内部流程为：

```text
actualPaInput ----------------------+
                                   |
feedbackCapture -> SigProc --------+-> aligned output/input pair
                                            |
                                            v
                                  chronological postinverse NLMS
                                            |
                                            v
                                  commit coefficients to predistorter
```

`SigProc`按整帧估计同步量，输出长度与PA输入参考一致；之后才开始逐样点更新。发送/反馈原始数组不做长度相等检查。

`paOutputFullScaleAmplitude` 位于签名末尾，默认1.0用于兼容旧Q1/FS1采集。当前 `PaModel` 与 `Channel` 的定点PA/FB输出默认FS2，因此直接训练其输出时应传2.0；近25 dBm链路若把PA、Channel和接收分析统一扩到FS4，则传4.0。PA输入与DPD输出仍是FS1。公共复增益补偿开启时，一个纯常数标尺误差可能被同步吸收，但仍应传真实量程；尤其设置 `enableComplexGainCompensation=False` 时，FS2输出若沿用默认1.0会被解码成一半幅度并使逐点误差和更新方向失真。

---

## 8. 严格样点提交模式

用于算法研究或硬件参考：

```python
dpdLms = DpdLms(
    parameters={
        "featureScaleMode": "running",
        "coefficientCommitMode": "sample",
        "updateDecimation": 1,
        "width": 0,
    }
)

dpdLms.BeginFrame()

for referenceSample, targetSample in sampleStream:
    predictedSample = dpdLms.UpdateSample(
        referenceSample,
        targetSample,
    )
```

运行尺度模式不需要事先提供完整帧。每个样点更新后，活动系数立即与影子系数一致。

风险是同一Wi-Fi帧内DPD系数随时间变化。真实发射建议：

```python
deploymentParameters = {"coefficientCommitMode": "frame"}
```

并在帧边界调用 `CommitCoefficients()`，或直接使用会自动帧末提交的 `UpdateFromLabels`、`UpdateIndirect`。

---

## 9. 定点逐样点接口

默认 `width=16`，因此 `UpdateSample`应接收整数I/Q码，而不是小于1的归一化浮点。该逐样点数字接口的参考、目标和DPD预测均为FS1；只有 `UpdateIndirect` 的PA/FB观测通过独立的 `paOutputFullScaleAmplitude` 解码：

```python
from inc.lib.DpdLms import DpdLms
from inc.utils.FixedPoint import FixedPoint


fixedPoint = FixedPoint(width=16)
referenceCodes = fixedPoint.EncodeComplex(referenceSignal)
targetCodes = fixedPoint.EncodeComplex(targetSignal)

dpdLms = DpdLms(
    parameters={
        "nonlinearOrders": (1, 3),
        "memoryDepth": 1,
        "crossMemoryDepth": 0,
        "featureScaleMode": "frame",
        "coefficientCommitMode": "frame",
        "width": 16,
    }
)

dpdLms.BeginFrame(referenceCodes)

for referenceCode, targetCode in zip(
    referenceCodes,
    targetCodes,
):
    predictedCode = dpdLms.UpdateSample(
        complex(referenceCode),
        complex(targetCode),
    )

dpdLms.CommitCoefficients()
predistortedCodes = dpdLms.Process(referenceCodes)
```

公开输入、返回预测和 `Process` 输出都是整数码值，容器类型仍为 `numpy.complex128`。内部特征、误差和系数更新保持浮点。

---

## 10. 活动映射和运行时修改

```python
from inc.lib.DpdLms import DpdLms


lmsParameters = {
    "learningRate": 0.05,
    "featureScaleMode": "frame",
    "width": 0,
}

dpdLms = DpdLms(parameters=lmsParameters)

lmsParameters["learningRate"] = 0.02
print(dpdLms.GetParameters()["learningRate"])
```

非结构参数在后续样点生效。`nonlinearOrders`、`memoryDepth` 或 `crossMemoryDepth`改变后，下一次同步结构会：

1. 重建特征顺序；
2. 恢复活动恒等系数；
3. 恢复影子恒等系数；
4. 重建历史和尺度数组；
5. 清除旧训练摘要。

这是必要的，因为旧系数数量和新特征含义可能不一致。

---

## 11. Benchmark

运行：

```powershell
python tests/BenchMark.py --dpd-lms
```

该场景比较：

- 批量岭回归在静态已知GMP标签上的精度；
- NLMS逐样点处理同一静态标签的精度；
- PA等效系数改变后，旧批量系数的失配；
- NLMS在一帧逐样点更新后的跟踪改善。

输出：

```text
results/dpd_lms_benchmark/dpd_lms_benchmark.csv
results/dpd_lms_benchmark/dpd_lms_benchmark.json
```

该Benchmark是算法结构和跟踪能力测试，不代替最终PA级Wi-Fi EVM/ACLR和双音IM3/IM5/IM7验证。

---

## 12. 常见错误

| 错误或现象 | 原因 | 处理 |
|---|---|---|
| `referenceSignal is required` | 帧尺度模式调用 `BeginFrame()` 时没有完整参考 | 传入参考帧或改用运行尺度 |
| 误把 `coefficientLearningRate` 当逐点步长 | 它属于批量父类 | 改用 `learningRate` |
| 活动系数没有在循环中变化 | 使用默认帧提交 | 循环后调用 `CommitCoefficients` |
| Wi-Fi ACLR在样点提交时变差 | 系数在OFDM帧内变化 | 使用帧提交 |
| 高阶系数更新极慢 | 训练峰值不足或尺度配置不当 | 检查 `PrepareFeatureScale` 输出 |
| 系数每次达到步进上限 | 步长过大、同步错误或异常权重 | 降低步长并检查反馈 |
| `onlineNmseDb`和`afterNmseDb`不同 | 在线NMSE使用时变系数 | 部署比较使用 `afterNmseDb` |
| 间接学习结果吸收反馈频响 | 反馈链未充分补偿或去嵌入 | 改善SigProc配置和反馈校准 |
| Python无法实时处理Wi-Fi采样率 | 逐样点循环是参考实现 | 移植到C、DSP或FPGA |
