# Fec 前向纠错原理与函数使用说明

## 1. 模块职责

`inc/lib/Fec.py` 集中保存前向纠错编码和译码函数。当前实现的是工程Wi-Fi解析描述使用的55/90短块LDPC码。

模块边界如下：

- `Fec.py`：校验矩阵构造、系统LDPC编码、软输入LDPC译码；
- `ParseWifi.py`：描述字段装包、导频插入、跨OFDM符号交织、信号均衡、字段语义解析；
- `WaveGenWifi.py`：把受保护的描述比特映射到VHT-SIG-A、HE-SIG-A或U-SIG位置。

该LDPC码用于提高工程描述字段中格式、MCS、空间结构和10 bit seed的恢复可靠性。它不是IEEE 802.11数据字段规定的标准LDPC码。

```mermaid
flowchart LR
    parameters["格式、MCS、空间结构、10 bit seed"] --> payload["ParseWifi：构造55 bit有效载荷"]
    payload --> encoder["Fec.EncodeDescriptorLdpc"]
    encoder --> codeword["90 bit LDPC码字"]
    codeword --> placement["ParseWifi：交织并插入14个导频"]
    placement --> channel["OFDM、PA和接收链"]
    channel --> equalization["ParseWifi：逐符号复增益补偿"]
    equalization --> decoder["Fec.DecodeDescriptorLdpc"]
    decoder --> recovered["恢复55 bit有效载荷"]
    recovered --> validation["ParseWifi：字段语义检查"]
```

**图示说明：**

1. FEC模块只处理二进制信息位或软码字，不感知OFDM、PA、带宽和采样率。
2. `ParseWifi` 负责把物理音调恢复为码字顺序，FEC译码器只接收90个实数软值。
3. 译码成功不仅要求硬判决，还要求全部35个奇偶校验方程成立。

---

## 2. 码长和码率

信息向量包含55 bit：

```math
\mathbf{m}
\in
\{0,1\}^{55}.
```

编码器增加35 bit校验信息，得到90 bit码字：

```math
\mathbf{c}
=
\begin{bmatrix}
\mathbf{m} \\
\mathbf{p}
\end{bmatrix},
\qquad
\mathbf{p}
\in
\{0,1\}^{35}.
```

码率为：

```math
R
=
\frac{55}{90}
\approx
0.611.
```

较低码率意味着90个发送bit中有35个用于冗余。接收端可以利用这些冗余约束修正PA非线性、噪声或符号间增益差异造成的部分错误。

---

## 3. 校验矩阵构造

### 3.1 矩阵结构

`BuildDescriptorLdpcMatrices()` 构造35乘90校验矩阵：

```math
\mathbf{H}
=
\begin{bmatrix}
\mathbf{A} & \mathbf{B}
\end{bmatrix}.
```

其中：

- $\mathbf{A}$ 的形状为35乘55，描述信息位与校验节点的连接；
- $\mathbf{B}$ 的形状为35乘35，是下双对角累加矩阵；
- 每个信息位连接三个校验节点。

累加矩阵可写成：

```math
\mathbf{B}
=
\begin{bmatrix}
1 & 0 & 0 & \cdots & 0 \\
1 & 1 & 0 & \cdots & 0 \\
0 & 1 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \ddots & \vdots \\
0 & 0 & \cdots & 1 & 1
\end{bmatrix}.
```

### 3.2 稀疏连接选择

对每一个信息位，内部函数 `TripleScore()` 从三个不同校验行组成的候选中选择连接位置。评分优先级为：

1. 避免重复使用已经出现过的校验行对；
2. 限制当前最大校验节点度数；
3. 平衡校验节点总度数；
4. 使用确定性循环次序打破相同评分。

如果两个信息列重复连接到同一对校验节点，Tanner图中会形成长度为4的短环。短环会让迭代消息过早相关，降低belief propagation或min-sum译码效果。因此构造器优先避免重复行对。

### 3.3 确定性和缓存

矩阵完全由固定维度和确定性规则构造，不调用随机数发生器。第一次调用后结果由函数缓存，后续编码和译码复用同一矩阵。

返回矩阵被设置为只读，避免调用方意外修改编码器和译码器共同依赖的校验关系。矩阵不是模块级数据全局变量。

---

## 4. 系统LDPC编码

合法码字必须满足：

```math
\mathbf{H}\mathbf{c}
=
\mathbf{0}
\bmod 2.
```

代入分块矩阵：

```math
\mathbf{A}\mathbf{m}
+
\mathbf{B}\mathbf{p}
=
\mathbf{0}
\bmod 2.
```

在二元域中，加法和减法相同，因此：

```math
\mathbf{B}\mathbf{p}
=
\mathbf{A}\mathbf{m}
\bmod 2.
```

先计算：

```math
\mathbf{q}
=
\mathbf{A}\mathbf{m}
\bmod 2.
```

由于 $\mathbf{B}$ 是下双对角矩阵，校验位可以递推得到：

```math
p_0=q_0,
```

```math
p_i
=
q_i
\mathbin{\mathrm{XOR}}
p_{i-1},
\qquad
i=1,\ldots,34.
```

最终码字前55 bit就是原始信息，后35 bit是校验信息，因此称为系统码：

```math
\mathbf{c}
=
\begin{bmatrix}
\mathbf{m} \\
\mathbf{p}
\end{bmatrix}.
```

`EncodeDescriptorLdpc()` 在返回前重新计算综合，任何非零综合都会触发内部编码错误。

---

## 5. 软输入normalized min-sum译码

### 5.1 软输入定义

发送端使用BPSK：

```math
x_i
=
1-2c_i.
```

因此：

```math
c_i=0
\Rightarrow
x_i=+1,
```

```math
c_i=1
\Rightarrow
x_i=-1.
```

`DecodeDescriptorLdpc()` 接受90个有限实数。正值表示更倾向bit 0，负值表示更倾向bit 1，绝对值表示可靠程度。

译码器使用非零软值幅度中位数 $\alpha$ 做尺度归一化：

```math
L_i^{(0)}
=
\frac{2r_i}{\alpha}.
```

然后把初始软值限制在有限范围内，避免极大值在迭代中造成数值不稳定。

### 5.2 校验节点更新

设变量节点 $v$ 发送到校验节点 $c$ 的消息为 $L_{v,c}$。归一化min-sum校验节点更新为：

```math
L_{c,v}
=
\beta
\left(
\prod_{u\ne v}
\mathrm{sign}(L_{u,c})
\right)
\min_{u\ne v}
\left|L_{u,c}\right|.
```

当前短码使用：

```math
\beta=0.5.
```

普通min-sum会高估校验节点输出幅度。系数0.5降低这种高估，并提高低度累加节点下的迭代稳定性。

### 5.3 变量节点更新

变量节点后验软值为：

```math
L_v^{\mathrm{post}}
=
L_v^{(0)}
+
\sum_c L_{c,v}.
```

发送回某一个校验节点时，需要去掉该校验节点刚发送来的信息：

```math
L_{v,c}
=
L_v^{\mathrm{post}}
-
L_{c,v}.
```

这样可以避免一条边上的消息立即反馈给自身。

### 5.4 硬判决和停止条件

硬判决规则为：

```math
\hat c_v
=
\begin{cases}
0, & L_v^{\mathrm{post}}\ge 0, \\
1, & L_v^{\mathrm{post}}<0.
\end{cases}
```

每轮计算综合：

```math
\mathbf{s}
=
\mathbf{H}\hat{\mathbf{c}}
\bmod 2.
```

只有：

```math
\mathbf{s}
=
\mathbf{0}
```

时才返回前55个系统信息位。默认最多迭代60轮；如果没有收敛到合法码字，函数抛出 `ValueError`，调用方不能把未通过校验的结果当作有效参数。

---

## 6. 函数接口

### 6.1 `BuildDescriptorLdpcMatrices()`

函数签名：

```python
from typing import Tuple

def BuildDescriptorLdpcMatrices() -> Tuple[np.ndarray, np.ndarray]:
    ...
```

返回值：

- 第一个数组：形状为 `(35, 90)` 的完整校验矩阵；
- 第二个数组：形状为 `(35, 55)` 的信息位子矩阵；
- 两个数组元素都是0或1，数据类型为 `numpy.uint8`；
- 返回数组是只读的。

### 6.2 `EncodeDescriptorLdpc(messageBits)`

输入要求：

- 一维或可展平为一维的NumPy数组；
- 恰好55个元素；
- 每个元素只能是0或1。

返回形状为 `(90,)` 的 `numpy.uint8` 系统码字。

### 6.3 `DecodeDescriptorLdpc(softCodeword, maximumIterations=60)`

输入要求：

- `softCodeword` 可展平为90个有限实数；
- 正值倾向bit 0，负值倾向bit 1；
- `maximumIterations` 必须是正整数。

成功时返回形状为 `(55,)` 的纠正后信息位。无法收敛时抛出 `ValueError`。

---

## 7. 最小编码和译码示例

```python
import numpy as np

from inc.lib.Fec import (
    BuildDescriptorLdpcMatrices,
    DecodeDescriptorLdpc,
    EncodeDescriptorLdpc,
)

randomGenerator = np.random.default_rng(101)
messageBits = randomGenerator.integers(
    0,
    2,
    size=55,
    dtype=np.uint8,
)

codeword = EncodeDescriptorLdpc(messageBits)

# Positive soft values represent bit zero and negative values represent bit one.
softCodeword = 1.0 - 2.0 * codeword.astype(float)
softCodeword += 0.15 * randomGenerator.standard_normal(90)

decodedBits = DecodeDescriptorLdpc(softCodeword)
assert np.array_equal(decodedBits, messageBits)

parityCheckMatrix, _ = BuildDescriptorLdpcMatrices()
syndrome = np.mod(
    parityCheckMatrix.astype(np.int64)
    @ codeword.astype(np.int64),
    2,
)
assert not np.any(syndrome)
```

示例中的软值已经是均衡后的BPSK判决量。实际Wi-Fi接收链中，这些值由 `ParseWifi` 完成导频复增益补偿和撤销交织后提供。

---

## 8. 人工错误测试示例

下面的例子在硬码字对应的软值中翻转少量位置：

```python
import numpy as np

from inc.lib.Fec import DecodeDescriptorLdpc, EncodeDescriptorLdpc

messageBits = np.zeros(55, dtype=np.uint8)
messageBits[::4] = 1
codeword = EncodeDescriptorLdpc(messageBits)

softCodeword = 1.0 - 2.0 * codeword.astype(float)
softCodeword[[2, 17, 44, 71]] *= -1.0

decodedBits = DecodeDescriptorLdpc(softCodeword)
assert np.array_equal(decodedBits, messageBits)
```

能够纠正多少错误不是一个固定bit数量，它取决于错误位置、软可靠度、Tanner图结构以及其他样点提供的信息。工程回归测试使用四个分散翻转验证当前描述字段路径，但这不表示任意四个错误都存在数学保证。

---

## 9. 异常处理示例

```python
import numpy as np

from inc.lib.Fec import DecodeDescriptorLdpc, EncodeDescriptorLdpc

try:
    EncodeDescriptorLdpc(np.zeros(54, dtype=np.uint8))
except ValueError as error:
    print(error)

try:
    DecodeDescriptorLdpc(
        np.zeros(90, dtype=float),
        maximumIterations=0,
    )
except ValueError as error:
    print(error)
```

编码器不会自动填充或截断信息位，译码器也不会返回未通过校验的近似结果。这样可以防止错误配置悄悄进入Wi-Fi参考波形重建。

---

## 10. Python版本兼容性

`Fec.py` 只依赖：

- Python标准库中的 `functools`、`itertools` 和 `typing`；
- 项目已有的NumPy。

代码不使用Python 3.10以后才支持的类型联合语法，也不依赖Cython、Numba或单独的LDPC二进制扩展，因此同一源码支持Python 3.9和Python 3.12。
