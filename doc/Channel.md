# Channel：PA到接收端链路模型

## 1. 模块边界

`inc/lib/Channel.py` 描述PA前多通道耦合、独立非线性PA、PA后耦合、外部仪表采样和板载反馈接收机。`sampleMode` 决定耦合后的PA输出进入哪条采样路径：

```mermaid
flowchart LR
    raw["用户原始波形"] --> calibration["Channel内部功率闭环<br/>调整PA输入缩放"]
    target["用户目标输出功率 dBm"] --> calibration
    calibration --> preCoupling["PA前耦合 Hpre(z)"]
    preCoupling --> pa["每路独立PA<br/>Wiener/GMP/Doherty"]
    pa --> paOutput["各PA自身输出 yPA(n)"]
    paOutput --> detector["有效突发功率检测"]
    detector -. "误差超出容差时反馈" .-> calibration
    paOutput --> postCoupling["PA后耦合 Hpost(z)"]
    postCoupling --> phase["公共固定相位旋转"]
    phase --> mode{"sampleMode"}
    mode -->|forward| instrument["前向仪表采样<br/>跳过fb专用非理想"]
    mode -->|fb| fbAnalog["板载反馈模拟链<br/>频响/增益/非线性/时频偏/IQ/DC"]
    instrument --> noise["AddNoise"]
    fbAnalog --> noise
    noise --> fbAdc{"fb模式？"}
    fbAdc -->|forward| receiver["仪表采样波形"]
    fbAdc -->|fb| adc["反馈ADC限幅与量化"]
    adc --> receiverFb["反馈采样波形"]
```

图示说明：

- `sampleMode="forward"` 是默认值，表示用校准仪表直接观测PA主路输出。所有 `fb...` 参数都被保留但不生效，因此不会把板载反馈接收机失真混入PA/DPD评价。
- `sampleMode="fb"` 表示通过板载反馈接收链采样；只有此模式会执行 `fbGainDb`、`fbFirTaps`、时延、CFO/SFO、I/Q不平衡、反馈非线性、限幅、DC和ADC量化。
- `prePaCouplingPaths` 在非线性之前把其他通道的延迟复泄漏叠加到每个PA输入；`postPaCouplingPaths` 在非线性之后混合各PA输出。两者都与forward/fb选择无关。
- `Channel.Process(rawSignal, outputPowerDbm=...)` 是推荐入口。用户只提供任意初始幅度的原始波形和目标PA输出功率；Channel内部调整PA输入、反复观测PA输出并收敛，然后只对最终PA输出执行一次所选采样路径。
- `Channel.Process(rawSignal)` 保留无功率校准的单次PA→采样路径，主要供ILC每轮plant调用。
- `Channel.ProcessPaOutput` 接收各PA已经产生但尚未经过输出耦合的矩阵，不再次运行PA，依次执行PA后耦合和所选采样路径。
- 功率闭环由Channel私有持有的 `PowerCalibration` 完成。普通用户不需要构造、配置或调用校准器。
- PA输出功率闭环只观察所有采样路径非理想之前的干净PA输出，因此不会把仪表或板载反馈接收机的失真误认为PA发射功率。
- 实验室“仪表闭环ILC”把forward模式的Channel作为plant；验证板载反馈算法时把fb模式的Channel作为plant，但最终EVM/ACLR仍应由独立forward模式采样评价。

### 1.1 前向仪表采样

前向模式以高性能VSA作为相对可信的黄金参考。设公共相位为 $\phi_c$，仪表噪声为 $w(n)$：

```math
z_{\mathrm{forward}}(n)
=
y_{\mathrm{PA}}(n)\exp(j\phi_c)+w(n).
```

`fb...` 配置不会进入该公式。这一点允许用同一组反馈非理想参数在forward和fb之间切换并进行公平对比，而不需要删除配置。

### 1.2 板载反馈采样

反馈模式的内部顺序为：

```text
PA output
  -> common phase
  -> feedback gain/phase and FIR
  -> third-order receiver distortion and envelope clipping
  -> fractional delay/SFO resampling and integer delay
  -> CFO phase ramp
  -> IQ gain/phase imbalance and DC offset
  -> AWGN
  -> feedback ADC clipping and quantization
```

用组合算子表示：

```math
z_{\mathrm{fb}}
=
Q_{\mathrm{ADC}}
\left\{
F_{\mathrm{IQ,DC}}
\left[
F_{\mathrm{time,freq}}
\left(
F_{\mathrm{NL}}
\left\{
H_{\mathrm{fb}}[y_{\mathrm{PA}}]
\right\}
\right)
\right]
+w
\right\}.
```

这条路径故意包含板载观察接收机的非理想。若直接用未经校准的 $z_{\mathrm{fb}}$ 更新ILC，算法可能学习PA与反馈链路的组合逆响应；因此工程测试应同时保留forward模式作为独立评价路径。

### 1.3 PA前与PA后多通道耦合

输入和输出矩阵都使用“样点数 × 物理通道数”。设第 $i$ 路原始输入为 $x_i(n)$，PA前耦合后第 $j$ 路实际激励为：

```math
u_j(n)
=
x_j(n)
+
\sum_{i\ne j}
\sum_{k=0}^{K_{\mathrm{pre}}-1}
h^{\mathrm{pre}}_{j,i}(k)x_i(n-k).
```

每一路再经过自己配置的Wiener、GMP或Doherty PA：

```math
y_j(n)=F_j\{u_j(n)\}.
```

PA后耦合为：

```math
v_j(n)
=
y_j(n)
+
\sum_{i\ne j}
\sum_{k=0}^{K_{\mathrm{post}}-1}
h^{\mathrm{post}}_{j,i}(k)y_i(n-k).
```

代码始终保留每路单位直通项，用户配置的路径作为额外串扰相加。每条路径使用：

```math
h_{j,i}
=
10^{G_{j,i}/20}
\exp(j\phi_{j,i})
f_{j,i}.
```

$f_{j,i}$ 是可选复FIR；随后再施加独立整数和分数时延。路径方向明确规定为 `sourceChain` 指向 `destinationChain`，因此0到1和1到0可以使用不同增益、相位、FIR及时延。

```mermaid
flowchart LR
    x0["x0"] --> sum0["PA0输入求和"]
    x1["x1"] --> sum1["PA1输入求和"]
    x0 --> c01["0→1<br/>增益/相位/FIR/时延"]
    c01 --> sum1
    x1 --> c10["1→0<br/>增益/相位/FIR/时延"]
    c10 --> sum0
    sum0 --> pa0["PA0"]
    sum1 --> pa1["PA1"]
    pa0 --> out0["输出0求和"]
    pa1 --> out1["输出1求和"]
    pa0 --> p01["0→1后耦合"]
    p01 --> out1
    pa1 --> p10["1→0后耦合"]
    p10 --> out0
```

**图示说明**：PA前耦合改变非线性工作点并产生跨通道包络相关失真；PA后耦合只混合已经生成的失真。当前模型描述线性耦合网络与独立非线性PA的级联。如果天线反射波会改变PA负载，使PA本身直接依赖其他通道输出，则还需要有源负载牵引模型，不能仅用PA后线性相加代替。

### 1.4 耦合条件下的联合功率校准

`outputPowerDbm=(P_0,P_1,...)` 仍表示PA后耦合之前每个物理PA自身的目标输出功率。只有PA前耦合时，调整一路输入会同时改变多个PA输出，Channel默认自动启用联合校准。

令隐藏驱动dB向量为 $\mathbf d$，实测PA功率向量为 $\mathbf p(\mathbf d)$。代码逐路增加 `calibrationProbeStepDb`，以有限差分估计：

```math
J_{m,n}
\approx
\frac{
p_m(\mathbf d+\Delta d\mathbf e_n)-p_m(\mathbf d)
}{
\Delta d
}.
```

联合更新为：

```math
\Delta\mathbf d
=
\left(
\mathbf J^T\mathbf J+\lambda\mathbf I
\right)^{-1}
\mathbf J^T
\left(
\mathbf p_{\mathrm{target}}-\mathbf p
\right).
```

`jointPowerCalibration=None` 表示自动模式：存在PA前耦合时启用Jacobian联合更新，否则沿用较快的逐链闭环。用户也可以显式设为 `True` 或 `False`。PA后耦合不进入功率检测，因此不会改变“每个PA自身输出dBm”的含义。

## 2. 固定相位旋转

令PA复包络输出为：

```math
y_{\mathrm{PA}}(n)=I(n)+jQ(n)
```

固定相位旋转为：

```math
y_{\phi}(n)=y_{\mathrm{PA}}(n)\exp(j\phi)
```

当前只支持：

```math
\phi\in\{-90^\circ,0^\circ,+90^\circ\}
```

三个取值可以直接写成I/Q交换关系：

```math
\phi=0^\circ:\quad I_{\phi}=I,\qquad Q_{\phi}=Q
```

```math
\phi=+90^\circ:\quad I_{\phi}=-Q,\qquad Q_{\phi}=I
```

```math
\phi=-90^\circ:\quad I_{\phi}=Q,\qquad Q_{\phi}=-I
```

因为复指数的模为1，所以理想相位旋转不改变瞬时幅度和平均功率：

```math
\left|y_{\phi}(n)\right|
=
\left|y_{\mathrm{PA}}(n)\right|
```

```math
\mathbb{E}\left[\left|y_{\phi}(n)\right|^2\right]
=
\mathbb{E}\left[\left|y_{\mathrm{PA}}(n)\right|^2\right]
```

相位旋转可用来模拟线缆电长度、本振初相、固定移相器或接收通道的公共相位差。它是线性影响，不会自行产生互调或频谱再生。

## 3. 白噪声模型

### 3.1 圆对称复高斯噪声

接收端波形为：

```math
r(n)=y_{\phi}(n)+w(n)
```

白噪声写成：

```math
w(n)=w_{\mathrm{I}}(n)+jw_{\mathrm{Q}}(n)
```

若配置的复包络总RMS为 `noiseAmpMv`，则I、Q两部分相互独立，并分别满足：

```math
w_{\mathrm{I}}(n),w_{\mathrm{Q}}(n)
\sim
\mathcal{N}\left(0,\frac{\sigma_w^2}{2}\right)
```

这里：

```math
\sigma_w
=
\sqrt{\mathbb{E}\left[\left|w(n)\right|^2\right]}
```

因此，`noiseAmpMv=10` 的含义是复包络总RMS电压为10 mV；不是I路10 mV再加Q路10 mV。每个实分量的RMS为：

```math
\sigma_{\mathrm{I}}
=
\sigma_{\mathrm{Q}}
=
\frac{10}{\sqrt{2}}\ \mathrm{mV}
```

白噪声在时间上独立同分布，理想功率谱密度在离散复基带Nyquist区间内为常数。它会抬高噪声底、恶化SNR和EVM，但不会像PA奇数阶非线性那样形成确定的IM3、IM5或IM7谱线。

### 3.2 由毫伏控制

`noiseAmpMv=A` 时，物理RMS电压为：

```math
V_{\mathrm{noise,rms}}=A\times 10^{-3}\ \mathrm{V}
```

在负载阻抗为 $R$ 时，对应噪声功率为：

```math
P_{\mathrm{noise,W}}
=
\frac{V_{\mathrm{noise,rms}}^2}{R}
```

```math
P_{\mathrm{noise,dBm}}
=
10\log_{10}
\left(
\frac{P_{\mathrm{noise,W}}}{10^{-3}}
\right)
```

例如 $R=50\ \Omega$、复包络RMS为10 mV：

```math
P_{\mathrm{noise,W}}
=
\frac{(10\times10^{-3})^2}{50}
=
2\times10^{-6}\ \mathrm{W}
```

```math
P_{\mathrm{noise,dBm}}
\approx
-26.99\ \mathrm{dBm}
```

### 3.3 由dBm控制

`noisePwrDbm=P` 时先换算为RMS电压：

```math
V_{\mathrm{noise,rms}}
=
\sqrt{R\times10^{-3}}\;10^{P/20}
```

所以在50 Ω系统中，`noiseAmpMv=10` 与 `noisePwrDbm=-26.99` 表示相同的噪声强度。`noiseAmpMv`、`noisePwrDbm` 和 `noiseSnrDb` 三个参数互斥：

- 三者都是 `None`：不加噪。
- 只有 `noiseAmpMv` 非 `None`：使用毫伏控制。
- 只有 `noisePwrDbm` 非 `None`：使用dBm控制。
- 只有 `noiseSnrDb` 非 `None`：使用有效突发信号与噪声的功率比控制。
- 任意两个或三个同时非 `None`：物理定义冲突，配置无效。

### 3.4 由SNR控制

`noiseSnrDb=S` 表示每一路有效突发信号功率与所加复白噪声功率之比为：

```math
S
=
10\log_{10}
\left(
\frac{P_{\mathrm{signal}}}
     {P_{\mathrm{noise}}}
\right).
```

因为复包络功率与RMS幅度平方成正比，所以每路噪声总RMS为：

```math
\sigma_{w,m}
=
x_{\mathrm{rms},m}
10^{-S/20},
```

其中 $x_{\mathrm{rms},m}$ 是第 $m$ 路移相后信号的有效区RMS。Channel复用PA功率校准的有效突发检测规则：

```math
x_{\mathrm{rms},m}
=
\sqrt{
\frac{
\sum_n M_m(n)\left|x_m(n)\right|^2
}{
\sum_n M_m(n)
}
}.
```

$M_m(n)$ 是该路有效样点掩码。它排除前后补零和长占空比关断区，只填充长度不超过 `activeGapToleranceSamples` 的短低幅空洞。因此，带有补零的30 dB配置仍表示开启期间约30 dB SNR，不会因为记录中有大量零而错误降低噪声。

SISO只有一个噪声RMS；MIMO对每一列独立计算 $x_{\mathrm{rms},m}$，因此幅度较小的链会得到相应较小的噪声RMS，但每路目标SNR相同。`noiseSnrDb` 允许任意有限dB值，包括负SNR。

### 3.5 物理电压到归一化波形

工程规定PA归一化输出RMS等于1时，对应 `maximumOutputPowerDbm`。满量程物理RMS电压为：

```math
V_{\mathrm{FS,rms}}
=
\sqrt{R\times10^{-3}}\;
10^{P_{\mathrm{max,dBm}}/20}
```

加入内部浮点波形的归一化噪声RMS为：

```math
\sigma_{\mathrm{noise,norm}}
=
\frac{V_{\mathrm{noise,rms}}}
       {V_{\mathrm{FS,rms}}}
```

这种换算保证相同的10 mV在浮点接口和16位定点接口中表示同一个物理噪声，而不是把“10”错误地当成归一化幅度或整数码。

## 4. 反馈链路非理想及其对ILC的影响

### 4.1 增益、相位和频率响应

反馈FIR及耦合增益为：

```math
v_m(n)
=
10^{G_{\mathrm{fb}}/20}
\exp(j\phi_{\mathrm{fb}})
\sum_{k=0}^{K-1}h_k y_m(n-k).
```

`fbFirTaps=None` 等价于 $h_0=1$。FIR按每一路独立卷积并保留原记录长度。若DPD不去嵌反馈频响，收敛条件可能变成 $H_{\mathrm{fb}}Y\approx X$，真实主路输出则错误地趋向 $Y\approx X/H_{\mathrm{fb}}$。

### 4.2 反馈接收机非线性和限幅

三阶反馈非线性为：

```math
v_{\mathrm{nl}}(n)
=
v(n)+c_3|v(n)|^2v(n).
```

`fbThirdOrderCoefficient` 是可为复数的 $c_3$，其实部主要改变AM-AM，虚部同时产生AM-PM。`fbClipAmplitude=A` 时再进行复包络径向限幅。若幅度没有超过门限：

```math
v_{\mathrm{clip}}(n)=v_{\mathrm{nl}}(n),
\qquad
|v_{\mathrm{nl}}(n)|\leq A.
```

若幅度超过门限，则只压缩幅度并保留相位：

```math
v_{\mathrm{clip}}(n)
=
A\frac{v_{\mathrm{nl}}(n)}
       {|v_{\mathrm{nl}}(n)|},
\qquad
|v_{\mathrm{nl}}(n)|>A.
```

反馈接收机压缩会让ILC补偿“PA+观察接收机”的组合非线性，因此最终必须使用forward仪表采样独立验证。

### 4.3 时延、CFO和SFO

分数时延与采样频偏使用保持长度的插值模型。正的 `fbFractionalDelaySamples=d` 取源位置 $n-d$；采样偏差为 $\epsilon_{\mathrm{sfo}}=\mathrm{ppm}\times10^{-6}$：

```math
v_{\mathrm{sfo}}(n)
=
v_{\mathrm{clip}}
\left(
n(1+\epsilon_{\mathrm{sfo}})-d
\right).
```

随后加入非负整数时延 $D$，并施加载波偏移：

```math
v_{\mathrm{cfo}}(n)
=
v_{\mathrm{sfo}}(n-D)
\exp
\left(
j2\pi\frac{\Delta f}{f_s}n
\right).
```

超出原采样记录的插值位置补零。`sampleRateHz` 决定CFO相位斜率的物理尺度。

### 4.4 I/Q不平衡与直流偏置

令I/Q增益不平衡为 $\Delta G$ dB、正交相位误差为 $\theta$：

```math
g_I=10^{\Delta G/40},
\qquad
g_Q=10^{-\Delta G/40}.
```

实现的实分量模型为：

```math
I'(n)=g_I I(n),
```

```math
Q'(n)
=
g_Q
\left[
\cos(\theta)Q(n)+\sin(\theta)I(n)
\right].
```

最后加复直流偏置：

```math
v_{\mathrm{IQ,DC}}(n)
=
I'(n)+jQ'(n)+d_{\mathrm{DC}}.
```

非零I/Q误差会形成共轭镜像分量，不能只靠公共复增益完全去除。

### 4.5 反馈ADC

`fbAdcWidth` 取 $W$ 时启用反馈ADC，`fbAdcFullScale` 取 $A_{\mathrm{FS}}$ 时定义每个I/Q分量的正负满量程。每个分量先转换成码值：

```math
q
=
\mathrm{clip}
\left(
\mathrm{round}
\left[
\frac{x}{A_{\mathrm{FS}}}2^{W-1}
\right],
-2^{W-1},
2^{W-1}-1
\right),
```

再解码回内部浮点采样：

```math
\widehat{x}
=
A_{\mathrm{FS}}\frac{q}{2^{W-1}}.
```

`fbAdcWidth=None` 表示不启用反馈ADC模型。这个位宽与Channel公开接口 `width` 不同：前者模拟板载反馈ADC内部量化，后者定义Python函数边界的I/Q码格式。

### 4.6 相位、噪声与ILC

固定相位是确定性线性项。若链路无噪声并且ILC执行公共复增益对齐，则公共相位通常可以被同步步骤准确估计：

```math
\widehat{g}
=
\frac{\boldsymbol{x}^{H}\boldsymbol{y}}
       {\boldsymbol{x}^{H}\boldsymbol{x}}
```

噪声是随机项，不能由确定性预失真完全消除。即使PA非线性已被很好补偿，EVM也会受到噪声底限制。简化地写：

```math
\mathrm{EVM}_{\mathrm{floor}}^2
\approx
\frac{\sigma_w^2}
     {\mathbb{E}[|x(n)|^2]}
```

当 `Channel` 直接作为ILC被控对象时，每一轮会得到新的独立噪声样本。这更接近真实反馈接收机，但也会使逐轮MSE或EVM出现随机波动。固定 `randomSeed` 只保证整次仿真的噪声序列可复现，并不让每轮重复同一段噪声；调用 `ResetRandomGenerator` 才会从序列起点重新开始。

## 5. 参数作用位置与可观测量示意图

本节的图片是“参数对应关系示意图”，不是参数遍历曲线，也不代表某一台真实仪表或芯片的指标极限。它们回答三个工程问题：

1. 参数作用在 PA 前、PA 后、前向采样还是反馈采样的哪个位置。
2. 改变参数后，最先应该在幅度、相位、时延、频谱、星座或功率闭环的哪个可观测量中寻找变化。
3. 哪些参数描述物理链路，哪些参数只控制数值求解、测量口径或公开定点接口。

### 5.1 参数在链路中的作用位置

![Channel 参数作用位置图](./images/channel_parameter/channel_parameter_location_map.png)

图示说明：

- 主信号从左向右依次经过功率校准、PA 前耦合、逐路 PA、PA 后耦合和公共固定移相。
- `sampleMode="forward"` 在公共移相后进入仪表前向采样支路，不经过任何 `fb...` 参数。
- `sampleMode="fb"` 才会继续经过反馈增益/FIR、非线性/限幅、时延/CFO/SFO、I/Q 不平衡/DC 和反馈 ADC。
- 三种白噪声配置位于所选采样支路的末端，因此它们描述接收或采样噪声，而不是 PA 本身的非线性。
- `width` 只定义公开输入输出的整数码位宽；图中所有物理模块仍在内部浮点域计算。
- 功率校准参数控制“如何寻找 PA 输入预设值”，不会改变 PA、耦合网络或反馈接收机的物理模型。

### 5.2 参数与可观测现象的对应关系

![Channel 参数到可观测现象的关系](./images/channel_parameter/channel_parameter_effects.png)

图中各分区的含义如下。

| 分区 | 参数组 | 最直接的可观测现象 |
|---|---|---|
| A | `sourceChain`、`destinationChain`、`gainDb`、`firTaps` | 耦合方向、耦合幅度、带内纹波和陷波 |
| B | 耦合路径 `phaseDegrees`、`integerDelaySamples`、`fractionalDelaySamples` | 中心相位和随频率变化的相位斜率 |
| C | `fbIqGainImbalanceDb`、`fbIqPhaseImbalanceDegrees`、`fbDcOffset` | 星座椭圆化、共轭镜像和星座中心偏移 |
| D | `fbThirdOrderCoefficient`、`fbClipAmplitude`、`fbAdcWidth`、`fbAdcFullScale` | AM/AM 弯曲、硬限幅和量化台阶 |
| E | 反馈时延、CFO、SFO 和 `sampleRateHz` | 波形横向平移、逐样点相位旋转和时间轴伸缩 |
| F | 有效突发检测与三种噪声配置 | 功率统计窗口、噪声底、SNR 和 EVM |
| G | 目标功率与校准求解参数 | 输出功率收敛速度、稳态误差和 MIMO 联合收敛性 |

### 5.3 调参方向与物理含义

下表中的“增大”均指参数数值增大；对于负 dB 耦合增益，`-20 dB` 大于 `-40 dB`，因此代表更强耦合。

| 参数 | 数值增大时的主要变化 | 不应误解为 |
|---|---|---|
| 耦合 `gainDb` | 泄漏幅度增大，非对角频响更接近主通道 | 不会单独产生额外时延 |
| 耦合 `phaseDegrees` | 整条泄漏路径发生固定复旋转 | 理想固定相位不会改变路径幅度 |
| `integerDelaySamples` | 路径后移整数样点，频域相位斜率变陡 | 理想时延不会制造幅频纹波 |
| `fractionalDelaySamples` | 增加不足一个样点的时延和连续相位斜率 | 不是简单的固定相位偏置 |
| `firTaps` | 决定频率选择性、记忆长度、纹波和可能的陷波 | 不能只用一个中心增益概括 |
| `sourceChain` / `destinationChain` | 改变泄漏的方向和 MIMO 拓扑 | 它们不表示耦合强度 |
| 公共 `phaseDegrees` | 所选观测信号整体旋转 `-90`、`0` 或 `90` 度 | 不改变功率、噪声或非线性 |
| `fbGainDb` | 反馈链路整体幅度变化 | 不等于 PA 增益变化 |
| `fbPhaseDegrees` | 反馈链路整体相位变化 | 不等于 PA AM/PM |
| `fbFirTaps` | 反馈链路出现幅频纹波和群时延变化 | 不属于 PA 记忆效应 |
| `fbIntegerDelaySamples` | 反馈采样整体后移整数样点 | 不改变原始 PA 输出 |
| `fbFractionalDelaySamples` | 增加分数样点时延 | 不等于 SFO |
| `fbCarrierFrequencyOffsetHz` | 每个样点累积相位，星座随时间旋转 | 不只是一个固定相位 |
| `fbSamplingFrequencyOffsetPpm` | 反馈时间轴逐渐伸缩，帧越长累计偏差越大 | 不只是一个固定延迟 |
| `fbIqGainImbalanceDb` | I/Q 两轴尺度不一致，镜像泄漏增强 | 不是公共复增益 |
| `fbIqPhaseImbalanceDegrees` | I/Q 正交性变差，星座倾斜并产生镜像 | 不是公共相位旋转 |
| `fbDcOffset` | 星座中心平移并在零频出现直流分量 | 不会随信号幅度同比缩放 |
| `fbThirdOrderCoefficient` | 反馈接收机三阶弯曲和互调增强 | 不应被训练器误认为 PA 三阶项 |
| `fbClipAmplitude` | 阈值减小时更早进入硬限幅 | 不是平滑的 PA 压缩 |
| `fbAdcWidth` | 位宽增大时量化步长减小、量化噪声下降 | 不会恢复已经被模拟限幅的信息 |
| `fbAdcFullScale` | 满量程增大时更不易削顶，但同位宽下量化步长变粗 | 不是“越大越精确” |
| `noiseAmpMv` | 毫伏 RMS 增大时噪声底上升 | 不能与另两种噪声控制同时启用 |
| `noisePwrDbm` | dBm 噪声功率增大时噪声底上升 | 其换算依赖 `loadResistanceOhm` |
| `noiseSnrDb` | 数值增大时添加的噪声减小，EVM 通常改善 | 它不是噪声功率本身 |
| `randomSeed` | 只改变或复现噪声样本实现 | 不改变理论噪声方差 |
| `sampleRateHz` | 改变 Hz、ppm、秒与样点之间的换算 | 不会自动改变 PA 或耦合增益 |
| `activePowerThresholdDb` | 阈值升高时有效功率统计更集中在突发高幅区 | 不是接收机检测灵敏度模型 |
| `activeGapToleranceSamples` | 数值增大时会闭合更长的突发内部低幅间隙 | 不会补回真实缺失样点 |
| `loadResistanceOhm` | 改变电压、瓦特和 dBm 的换算关系 | 不改变归一化样本本身 |
| `maximumOutputPowerDbm` | 改变归一化 PA 输出 RMS 等于 1 时代表的物理功率 | 不是每次调用的目标功率 |
| `calibrationToleranceDb` | 数值增大时更容易提前停止，但允许更大功率误差 | 不改变目标值 |
| `maximumCalibrationIterations` | 数值增大时允许更多闭环尝试 | 不保证病态耦合下一定收敛 |
| `calibrationLearningRate` | 数值增大时单轮校正更激进 | 过大可能来回振荡 |
| `maximumDriveAdjustmentDb` | 数值增大时单轮允许更大的驱动修正 | 不会提高 PA 的物理饱和功率 |
| `jointPowerCalibration` | 在 MIMO 耦合下选择联合或逐链求解策略 | 它不是耦合开关 |
| `calibrationProbeStepDb` | 增大时 Jacobian 探测更明显，但局部线性近似变粗 | 不是实际输出功率步进 |
| `calibrationRegularization` | 增大时联合求解更稳定、更新更保守 | 过大可能留下功率偏差 |
| `width` | `0` 为物理浮点幅值；正整数使用对应满量程整数 I/Q 码 | 内部 PA 和信道计算仍是浮点 |

调用 `Process(inputSignal, outputPowerDbm=...)` 时，`outputPowerDbm` 才是本次运行希望每路 PA 达到的实际输出功率；它不是构造参数。图中 G 区把它作为闭环目标单独画出，是为了避免与 `maximumOutputPowerDbm` 混淆。

## 6. 参数表

构造接口：

```python
Channel(
    paModel=None,
    parameters=None,
    width=None,
    **parameterOverrides,
)
```

| 参数 | 默认值 | 单位 | 说明 |
|---|---:|---|---|
| `sampleMode` | `"forward"` | 无 | `"forward"`为校准仪表采样；`"fb"`为板载反馈接收机采样 |
| `sampleRateHz` | `1.0` | sample/s | CFO物理相位斜率使用的采样率；配置Hz级CFO时应设置真实值 |
| `phaseDegrees` | `0` | degree | 仅允许 `-90`、`0`、`90` |
| `noiseAmpMv` | `None` | mV RMS | 复包络总RMS噪声幅度 |
| `noisePwrDbm` | `None` | dBm | 配置端口上的总噪声功率 |
| `noiseSnrDb` | `None` | dB | 每路有效突发信号功率与复噪声功率之比 |
| `fbGainDb` | `0.0` | dB | fb模式反馈耦合与接收机电压增益 |
| `fbPhaseDegrees` | `0.0` | degree | fb模式附加反馈相位，允许任意有限值 |
| `fbFirTaps` | `None` | 无 | fb模式因果复FIR；`None`等价于单个单位抽头 |
| `fbIntegerDelaySamples` | `0` | sample | fb模式非负整数时延 |
| `fbFractionalDelaySamples` | `0.0` | sample | fb模式分数时延，范围 `[-0.5, 0.5)` |
| `fbCarrierFrequencyOffsetHz` | `0.0` | Hz | fb模式载波频偏 |
| `fbSamplingFrequencyOffsetPpm` | `0.0` | ppm | fb模式采样频偏，绝对值必须小于一百万 |
| `fbIqGainImbalanceDb` | `0.0` | dB | fb模式I/Q分量增益比 |
| `fbIqPhaseImbalanceDegrees` | `0.0` | degree | fb模式正交相位误差 |
| `fbDcOffset` | `0+0j` | normalized | fb模式复直流偏置 |
| `fbThirdOrderCoefficient` | `0+0j` | normalized | fb模式接收机三阶复多项式系数 |
| `fbClipAmplitude` | `None` | normalized | fb模式复包络径向限幅；正数启用 |
| `fbAdcWidth` | `None` | bit/I或Q | fb模式内部ADC位宽，支持2至32；`None`禁用 |
| `fbAdcFullScale` | `1.0` | normalized | fb模式内部ADC每个I/Q分量满量程 |
| `prePaCouplingPaths` | `None` | 无 | PA前串扰路径序列；每项配置源/目标链、增益、相位、FIR和时延 |
| `postPaCouplingPaths` | `None` | 无 | PA后串扰路径序列；格式与PA前路径相同 |
| `loadResistanceOhm` | `50.0` | Ω | dBm与RMS电压换算阻抗 |
| `maximumOutputPowerDbm` | `25.0` | dBm | 归一化PA输出RMS等于1所代表的功率 |
| `calibrationToleranceDb` | `0.25` | dB | 内部闭环允许的最大PA输出功率误差 |
| `maximumCalibrationIterations` | `60` | 次 | 内部闭环最多激励和测量PA的次数 |
| `calibrationLearningRate` | `0.8` | 无 | 尚未括住目标时的dB域修正比例 |
| `maximumDriveAdjustmentDb` | `6.0` | dB/次 | 单轮PA输入预设最大调整量 |
| `jointPowerCalibration` | `None` | 无 | `None`按PA前耦合自动选择；布尔值强制联合或逐链校准 |
| `calibrationProbeStepDb` | `0.05` | dB | 联合校准估计功率Jacobian的逐路扰动 |
| `calibrationRegularization` | `1e-6` | 无 | 联合校准正规方程的正则化系数 |
| `activePowerThresholdDb` | `-60.0` | dB | 相对峰值的有效突发功率门限 |
| `activeGapToleranceSamples` | `16` | sample | 有效区内部允许闭合的短低幅空洞 |
| `randomSeed` | `1701` | 无 | 非负整数可复现；`None` 使用系统熵 |
| `width` | `16` | bit/I或Q | `0` 为浮点，正整数为公开定点码位宽 |

与其他主类相同，默认值都定义在构造函数内部，并通过 `ChainMap` 与调用方配置合并。未知参数会产生警告、被忽略，并且不会中止处理；已识别但数值非法的参数仍会抛出异常。

`prePaCouplingPaths` 与 `postPaCouplingPaths` 中每个路径映射支持：

| 路径字段 | 默认值 | 说明 |
|---|---:|---|
| `sourceChain` | `0` | 泄漏源物理通道索引 |
| `destinationChain` | `1` | 泄漏进入的目标通道索引；不能等于源索引 |
| `gainDb` | `-30.0` | 电压耦合增益，使用20对数换算 |
| `phaseDegrees` | `0.0` | 路径固定相位 |
| `integerDelaySamples` | `0` | 非负因果整数时延 |
| `fractionalDelaySamples` | `0.0` | 分数时延，范围 `[-0.5,0.5)` |
| `firTaps` | `None` | 可选非空有限复FIR；`None`为单位抽头 |

路径中的未知字段会单独发出警告并忽略；已识别但非法的索引、时延、增益或FIR仍然报错。

主要接口为：

| 方法 | 参数 | 返回值或作用 |
|---|---|---|
| `Process(inputSignal, outputPowerDbm=None)` | 原始公开波形；可选共同目标dBm或逐链序列 | 有目标时内部闭环校准PA输入，随后执行 `sampleMode` 选择的采样路径；`None`时只执行一次PA与采样 |
| `CalibratePaInput(inputSignal, outputPowerDbm)` | 原始波形、目标功率 | 高级诊断入口；只运行内部PA输入闭环并返回收敛PA输入 |
| `GetLastPaInput()` | 无 | 返回最近一次内部闭环实际送入PA的波形 |
| `GetLastPaOutput()` | 无 | 返回最近一次内部闭环接受的干净PA输出 |
| `GetLastCalibrationMetrics()` | 无 | 返回目标、实测dBm、误差和迭代次数字典 |
| `ProcessPaOutput(paOutputSignal)` | 已有各PA自身输出 | 不运行PA或功率闭环，执行PA后耦合及所选采样路径 |
| `ResolveCouplingPaths(parameterName, chainCount=None)` | 路径参数名、可选链数 | 过滤未知子键并规范、校验耦合路径 |
| `ApplyCouplingPath(sourceSignal, couplingPath)` | 单路源信号、规范路径 | 应用FIR、整数/分数时延、增益与相位 |
| `ApplyMimoCoupling(inputSignal, parameterName)` | 多路矩阵、路径参数名 | 保留直通并累加所有非对角耦合 |
| `ApplyPrePaCoupling(inputSignal)` | PA前矩阵 | 生成每个PA真正看到的耦合激励 |
| `ApplyPostPaCoupling(paOutputSignal)` | 各PA自身输出矩阵 | 在采样前混合PA非线性输出 |
| `HasPrePaCoupling()` | 无 | 判断自动联合功率校准是否需要启用 |
| `ProcessBoundPaFloating(inputSignal)` | 实际PA激励 | 只运行绑定PA，不加耦合或采样影响 |
| `ProcessPaBankForCalibration(inputSignal)` | 公开试探波形 | 执行PA前耦合和PA，用于功率闭环 |
| `ApplyFeedbackLinearResponse(inputSignal)` | fb模拟输入 | 应用反馈增益、相位和FIR |
| `ApplyFeedbackNonlinearity(inputSignal)` | fb线性输出 | 应用反馈三阶非线性和包络限幅 |
| `ApplyFeedbackTimingAndFrequency(inputSignal)` | fb模拟波形 | 应用分数/整数时延、CFO和SFO |
| `ApplyFeedbackIqImbalance(inputSignal)` | fb时频偏输出 | 应用I/Q增益/相位误差和DC |
| `ApplyFeedbackAdc(inputSignal)` | fb含噪波形 | 应用反馈ADC分量限幅与量化 |
| `ApplyFeedbackAnalogImpairments(inputSignal)` | 公共移相后的PA输出 | 按固定顺序组合全部fb模拟非理想 |
| `FeedbackDirectSmallSignalGain()` | 无 | 返回fb线性直通分量的复小信号系数 |
| `ResolveSnrNoiseRmsPerChain(inputSignal)` | 内部归一化SISO/MIMO信号 | 返回按有效突发SNR推导的逐链复噪声总RMS |
| `ResetRandomGenerator()` | 无 | 按当前种子重放接收噪声序列 |

## 7. 典型使用方式

先根据已有信号选择入口：

| 用户已有数据或目标 | 推荐入口 | 是否再次运行PA |
|---|---|---|
| 校准仪表采样PA主路 | `sampleMode="forward"` | 取决于Process入口 |
| 板载反馈接收机采样 | `sampleMode="fb"`并配置 `fb...` 参数 | 取决于Process入口 |
| 原始波形和目标PA输出功率 | `Channel.Process(rawSignal, outputPowerDbm=20.0)` | 内部闭环多次，收敛后返回 |
| MIMO原始矩阵和逐链目标 | `Channel.Process(rawMatrix, outputPowerDbm=(22.0, 21.0))` | 有PA前耦合时联合闭环 |
| 已有精确PA输入，不需要设定功率 | `Channel.Process(paInputSignal)` | 一次 |
| 已有PA输出或仪器PA采集 | `Channel.ProcessPaOutput(paOutputSignal)` | 否 |
| 只验证移相 | `ProcessPaOutput`，三个噪声参数均为 `None` | 否 |
| 按毫伏添加接收噪声 | `noiseAmpMv` 非 `None` | 取决于所选入口 |
| 按端口功率添加接收噪声 | `noisePwrDbm` 非 `None` | 取决于所选入口 |
| 按有效突发SNR添加接收噪声 | `noiseSnrDb` 非 `None` | 取决于所选入口 |
| I/Q是定点整数码 | 所有模块使用相同正 `width` | 取决于所选入口 |
| MIMO samples×chains矩阵 | `Process` 绑定 `MimoPaModel` | 是 |

### 7.1 只验证固定移相

不绑定PA时仍可使用 `ProcessPaOutput`。下面把一段已知PA输出旋转90度，不添加噪声：

```python
import numpy as np

from inc.lib.Channel import Channel


paOutputSignal = np.array(
    [0.20 + 0.30j, -0.40 + 0.10j],
    dtype=np.complex128,
)
channel = Channel(
    parameters={
        "phaseDegrees": 90,
        "noiseAmpMv": None,
        "noisePwrDbm": None,
        "noiseSnrDb": None,
        "width": 0,
    }
)
receivedSignal = channel.ProcessPaOutput(paOutputSignal)

assert np.allclose(receivedSignal, 1j * paOutputSignal)
```

若 `phaseDegrees=-90`，结果为 `-1j * paOutputSignal`；若为0，则在无噪声条件下返回等值副本。

### 7.2 完整PA到前向仪表采样链路

```python
import numpy as np

from inc.lib.Channel import Channel
from inc.lib.PaModel import PaModel


paInputSignal = np.array(
    [0.05 + 0.02j, 0.20 - 0.10j, -0.35 + 0.25j],
    dtype=np.complex128,
)
paModel = PaModel(
    parameters={
        "modelName": "gmp",
        "width": 0,
    }
)
channel = Channel(
    paModel=paModel,
    parameters={
        "sampleMode": "forward",
        "sampleRateHz": 80.0e6,
        "phaseDegrees": 90,
        "noiseAmpMv": 10.0,
        "noisePwrDbm": None,
        "noiseSnrDb": None,
        "loadResistanceOhm": 50.0,
        "maximumOutputPowerDbm": 25.0,
        "randomSeed": 1019,
        "width": 0,
    },
)

receivedSignal = channel.Process(paInputSignal)
```

`Process` 内部只跨越一次公开数据边界，实际顺序为：

```mermaid
flowchart LR
    publicInput["公开PA输入"] --> decode["解码到内部浮点"]
    decode --> pa["PA.ProcessFloating"]
    pa --> phase["固定移相"]
    phase --> forward["forward仪表采样<br/>跳过全部fb参数"]
    forward --> noise["AddNoise"]
    noise --> encode["编码到公开输出"]
```

图示说明：`width=0` 时解码和编码是等值复制；`width>0` 时公开I/Q为整数码，但PA、移相和噪声仍在内部归一化浮点域计算。

### 7.3 前向仪表与板载反馈路径对比

推荐用两个Channel共享同一个PA模型：fb路径作为ILC训练观测，forward路径作为独立黄金参考和最终评价。已经得到一次干净PA输出时，用 `ProcessPaOutput` 分路可以避免PA被重复运行：

```python
from inc.lib.Channel import Channel


forwardChannel = Channel(
    paModel=paModel,
    parameters={
        "sampleMode": "forward",
        "sampleRateHz": 160.0e6,
        "noiseSnrDb": 50.0,
        "width": 0,
    },
)
feedbackChannel = Channel(
    paModel=paModel,
    parameters={
        "sampleMode": "fb",
        "sampleRateHz": 160.0e6,
        "fbGainDb": -6.0,
        "fbPhaseDegrees": 18.0,
        "fbFirTaps": (
            1.0 + 0.0j,
            0.08 - 0.04j,
            -0.02 + 0.01j,
        ),
        "fbIntegerDelaySamples": 12,
        "fbFractionalDelaySamples": 0.18,
        "fbCarrierFrequencyOffsetHz": 2500.0,
        "fbSamplingFrequencyOffsetPpm": 8.0,
        "fbIqGainImbalanceDb": 0.35,
        "fbIqPhaseImbalanceDegrees": 1.5,
        "fbDcOffset": 0.002 - 0.001j,
        "fbThirdOrderCoefficient": -0.015 + 0.004j,
        "fbClipAmplitude": 0.95,
        "fbAdcWidth": 14,
        "fbAdcFullScale": 1.0,
        "noiseSnrDb": 35.0,
        "width": 0,
    },
)

cleanPaOutput = paModel.Process(paInputSignal)
instrumentCapture = forwardChannel.ProcessPaOutput(cleanPaOutput)
feedbackCapture = feedbackChannel.ProcessPaOutput(cleanPaOutput)
```

`instrumentCapture` 用于最终EVM、ACLR和功率判断；`feedbackCapture` 用于训练或验证反馈校准。不能只用未经校准的 `feedbackCapture` 同时训练和评价DPD，否则容易得到“板载反馈EVM很好但主路仪表EVM变差”的虚假结论。

运行时也可以保留一套参数并切换：

```python
feedbackChannel.UpdateParameters(sampleMode="forward")
forwardCaptureFromSameObject = feedbackChannel.ProcessPaOutput(
    cleanPaOutput
)
```

forward模式会完整跳过所有 `fb...` 非理想，但公共 `phaseDegrees` 和三种互斥噪声控制仍然生效。

### 7.4 用户只提供原始波形与目标输出功率

普通用户不需要主动创建 `PowerCalibration`，也不需要先归一化原始波形。把任意初始幅度的波形和目标dBm直接交给 `Channel.Process`：

```python
from inc.lib.Channel import Channel
from inc.lib.PaModel import PaModel
from inc.lib.WaveGenWifi import WaveGenWifi


wifiWaveform = WaveGenWifi(
    parameters={
        "frameFormat": "EHT",
        "bandwidthMhz": 20,
        "mcs": 7,
        "numDataSymbols": 6,
        "sampleRateHz": 80.0e6,
        "width": 0,
    }
).Generate()
paModel = PaModel(
    parameters={"modelName": "wiener", "width": 0}
)
channel = Channel(
    paModel=paModel,
    parameters={
        "sampleMode": "forward",
        "sampleRateHz": wifiWaveform.sampleRateHz,
        "phaseDegrees": 0,
        "noiseAmpMv": 10.0,
        "noisePwrDbm": None,
        "noiseSnrDb": None,
        "maximumOutputPowerDbm": 25.0,
        "loadResistanceOhm": 50.0,
        "randomSeed": 1019,
        "width": 0,
    },
)
receivedSignal = channel.Process(
    wifiWaveform.samples,
    outputPowerDbm=20.0,
)

# Optional diagnostics remain available without exposing the drive preset.
referenceSignal = channel.GetLastPaInput()
cleanPaOutput = channel.GetLastPaOutput()
calibrationMetrics = channel.GetLastCalibrationMetrics()
print(calibrationMetrics)
```

对应流程为：

```mermaid
flowchart TD
    original["用户原始波形<br/>无需预归一化"] --> process["Channel.Process"]
    target["用户目标 20 dBm"] --> process
    process --> calibration["内部PowerCalibration<br/>隐藏输入缩放预设"]
    calibration --> pa["PA"]
    pa --> detector["有效突发功率检测"]
    detector --> decision{"误差在容差内？"}
    decision -->|否| calibration
    decision -->|是| cleanOutput["缓存干净PA输出"]
    cleanOutput --> phase["固定移相"]
    phase --> noise["AddNoise"]
    noise --> receiver["返回接收波形"]
```

图示说明：闭环每次都重新缩放原始波形并真实运行PA，不是在PA输出后乘常数伪造功率。有效区检测会排除前后补零和长静默。功率误差只由干净PA输出决定；相位和接收噪声在收敛后只执行一次，不会改变隐藏的PA输入预设。

### 7.5 毫伏、dBm和SNR三种噪声配置

在50 Ω端口上，10 mV复包络总RMS约等于 `-26.9897 dBm`。使用相同随机种子时，下面两个Channel会产生同一段噪声：

```python
import numpy as np

from inc.lib.Channel import Channel


zeroSignal = np.zeros(10000, dtype=np.complex128)
amplitudeChannel = Channel(
    parameters={
        "noiseAmpMv": 10.0,
        "noisePwrDbm": None,
        "noiseSnrDb": None,
        "loadResistanceOhm": 50.0,
        "maximumOutputPowerDbm": 25.0,
        "randomSeed": 73,
        "width": 0,
    }
)
powerChannel = Channel(
    parameters={
        "noiseAmpMv": None,
        "noisePwrDbm": -26.989700043360187,
        "noiseSnrDb": None,
        "loadResistanceOhm": 50.0,
        "maximumOutputPowerDbm": 25.0,
        "randomSeed": 73,
        "width": 0,
    }
)
noiseFromAmplitude = amplitudeChannel.ProcessPaOutput(
    zeroSignal
)
noiseFromPower = powerChannel.ProcessPaOutput(zeroSignal)

assert np.allclose(noiseFromAmplitude, noiseFromPower)
print(amplitudeChannel.ResolveNoiseRmsVolts())
print(amplitudeChannel.ResolveNoiseRmsNormalized())
```

`ResolveNoiseRmsVolts()` 返回0.01 V；`ResolveNoiseRmsNormalized()` 返回加入内部PA归一化波形的噪声RMS。

SNR方式不需要端口电压或满量程功率，它根据当前信号有效区计算噪声：

```python
activeSignal = np.exp(
    1j * 2.0 * np.pi * np.arange(100000) / 37.0
)
paddedSignal = np.concatenate(
    (
        np.zeros(1000, dtype=np.complex128),
        activeSignal,
        np.zeros(2000, dtype=np.complex128),
    )
)
snrChannel = Channel(
    parameters={
        "noiseAmpMv": None,
        "noisePwrDbm": None,
        "noiseSnrDb": 30.0,
        "randomSeed": 83,
        "width": 0,
    }
)
snrOutput = snrChannel.ProcessPaOutput(paddedSignal)
```

这里的30 dB针对 `activeSignal` 所在的开启区间。前1000个和后2000个零样点不会降低信号RMS，也不会导致噪声被错误配置得过小。实际使用时三个噪声参数只能选择一个。

### 7.6 Channel输出直接送入Analysis

继续使用7.4节得到的 `referenceSignal`、`wifiWaveform` 和 `receivedSignal`：

```python
from inc.lib.Analysis import Analysis


resultAnalysis = Analysis(
    referenceSignal,
    wifiWaveform,
    parameters={
        "maximumOutputPowerDbm": 25.0,
        "loadResistanceOhm": 50.0,
        "width": 0,
    },
)
metrics = resultAnalysis.Analyze(receivedSignal)

print(metrics["outputPowerDbm"])
print(metrics["snrDb"])
print(metrics["evmDb"], metrics["evmPercent"])
print(metrics["aclrWorstDb"])
```

固定移相会由Analysis的公共复增益步骤补偿；随机噪声、PA非线性和记忆失真仍会进入SNR和EVM残差。`Analysis` 的独立使用方式见 [Analysis.md §11](./Analysis.md)。

### 7.7 16位定点接口

定点模式下，输入输出容器仍是 `numpy.complex128`，但每个I/Q分量都是整数码：

```python
import numpy as np

from inc.lib.Channel import Channel
from inc.lib.PaModel import PaModel
from inc.utils.FixedPoint import FixedPoint


fixedPoint = FixedPoint(16)
floatingInput = np.array(
    [0.10 + 0.20j, -0.25 + 0.15j],
    dtype=np.complex128,
)
fixedInput = fixedPoint.EncodeComplex(floatingInput)
paModel = PaModel(
    parameters={"modelName": "wiener", "width": 16}
)
channel = Channel(
    paModel=paModel,
    parameters={
        "phaseDegrees": -90,
        "noiseAmpMv": 10.0,
        "maximumOutputPowerDbm": 25.0,
        "randomSeed": 91,
        "width": 16,
    },
)
fixedOutput = channel.Process(fixedInput)

assert fixedOutput.dtype == np.complex128
assert np.array_equal(
    fixedOutput.real, np.rint(fixedOutput.real)
)
assert np.array_equal(
    fixedOutput.imag, np.rint(fixedOutput.imag)
)
```

10 mV不会被直接当成整数码10。Channel先把物理电压转换为内部归一化RMS，生成浮点噪声，最后统一编码为16位整数码。

### 7.8 运行时更新参数和复现噪声

调用方传入的 `parameters` 字典保持活动状态；修改已识别键后，下一次处理会读取新值。`UpdateParameters` 可写入最高优先级覆盖：

```python
import numpy as np

from inc.lib.Channel import Channel


channelParameters = {
    "phaseDegrees": 0,
    "noiseAmpMv": None,
    "noisePwrDbm": None,
    "randomSeed": 101,
    "width": 0,
}
channel = Channel(parameters=channelParameters)
paOutputSignal = np.ones(1024, dtype=np.complex128)

unchangedSignal = channel.ProcessPaOutput(paOutputSignal)

# A live caller mapping changes the next channel evaluation.
channelParameters["phaseDegrees"] = 90
rotatedSignal = channel.ProcessPaOutput(paOutputSignal)

# UpdateParameters creates a higher-priority local override.
channel.UpdateParameters(noiseAmpMv=5.0)
channel.ResetRandomGenerator()
firstNoisySignal = channel.ProcessPaOutput(paOutputSignal)
channel.ResetRandomGenerator()
repeatedNoisySignal = channel.ProcessPaOutput(paOutputSignal)

assert np.allclose(unchangedSignal, paOutputSignal)
assert np.allclose(rotatedSignal, 1j * paOutputSignal)
assert np.array_equal(
    firstNoisySignal, repeatedNoisySignal
)
```

固定种子保证整次仿真的噪声序列可复现，但连续两次 `Process` 默认会消耗不同的随机样值。只有调用 `ResetRandomGenerator()` 才会从同一种子起点重放。

### 7.9 MIMO矩阵

Channel保留 `samples × chains` 形状，并对每个元素加入独立白噪声。绑定 `MimoPaModel` 后可以处理完整多链矩阵：

```python
from inc.lib.Channel import Channel
from inc.lib.PaModel import DohertyConfig, MimoPaModel
from inc.lib.WaveGenWifi import WaveGenWifi


wifiWaveform = WaveGenWifi(
    parameters={
        "frameFormat": "EHT",
        "bandwidthMhz": 20,
        "mcs": 5,
        "numDataSymbols": 4,
        "sampleRateHz": 80.0e6,
        "numTransmitAntennas": 2,
        "numSpatialStreams": 2,
        "spatialMapping": "dft",
        "width": 0,
    }
).Generate()
mimoPaModel = MimoPaModel(
    parameters={
        "numTransmitChains": 2,
        "paParametersPerChain": (
            {
                "modelName": "doherty",
                "dohertyConfig": DohertyConfig(
                    peakingTurnOnAmplitude=0.45,
                    peakingTransitionWidth=0.15,
                ),
            },
            {"modelName": "gmp"},
        ),
        "inputPowerDbPerChain": (0.0, -1.0),
        "outputPowerDbPerChain": (0.0, -0.5),
        "width": 0,
    }
)
channel = Channel(
    paModel=mimoPaModel,
    parameters={
        "sampleMode": "forward",
        "sampleRateHz": wifiWaveform.sampleRateHz,
        "prePaCouplingPaths": (
            {
                "sourceChain": 0,
                "destinationChain": 1,
                "gainDb": -28.0,
                "phaseDegrees": 20.0,
                "integerDelaySamples": 2,
                "fractionalDelaySamples": 0.15,
                "firTaps": (1.0 + 0.0j, 0.06 - 0.02j),
            },
            {
                "sourceChain": 1,
                "destinationChain": 0,
                "gainDb": -31.0,
                "phaseDegrees": -15.0,
                "integerDelaySamples": 4,
            },
        ),
        "postPaCouplingPaths": (
            {
                "sourceChain": 0,
                "destinationChain": 1,
                "gainDb": -23.0,
                "phaseDegrees": 35.0,
                "integerDelaySamples": 1,
            },
        ),
        "phaseDegrees": 0,
        "noiseAmpMv": None,
        "noisePwrDbm": None,
        "noiseSnrDb": 30.0,
        "randomSeed": 401,
        "width": 0,
    },
)
receivedMatrix = channel.Process(
    wifiWaveform.samples,
    outputPowerDbm=(22.0, 20.0),
)

assert receivedMatrix.shape == wifiWaveform.samples.shape
print(channel.GetLastCalibrationMetrics())
```

`outputPowerDbm=(22.0, 20.0)` 按列控制两个PA在输出耦合之前的自身功率。第一路是Doherty，第二路是GMP。PA前存在双向、不对称且不同时延的耦合，所以Channel自动使用联合Jacobian功率闭环；PA后再增加0到1的输出泄漏。传一个标量时，所有链使用同一目标。

当前公共 `phaseDegrees` 和fb接收机参数仍由所有链共用；耦合路径自身的增益、相位、FIR及时延则可以逐方向独立配置。噪声样值在各链之间独立，`noiseSnrDb` 按各路有效信号RMS分别设置强度。

## 8. `SmallestSISO.py`中的设置

最小SISO示例使用：

```python
channel = Channel(
    paModel=paModel,
    parameters={
        "sampleMode": "forward",
        "sampleRateHz": waveform.sampleRateHz,
        "phaseDegrees": 0,
        "noiseAmpMv": 10.0,
        "noisePwrDbm": None,
        "loadResistanceOhm": 50.0,
        "maximumOutputPowerDbm": 25.0,
        "randomSeed": 1019,
        "width": width,
    },
)
```

其执行边界为：

1. 示例直接调用 `channel.Process(waveform.samples, outputPowerDbm=20.0)`；调用方不创建功率校准器。
2. `Channel` 内部反复调整PA输入，直到干净PA输出达到目标，再执行0度移相和10 mV接收噪声。
3. `channel.GetLastPaInput()` 返回收敛的PA输入，作为ILC参考；隐藏的dB预设不对用户开放。
4. `RunFrequencyDomainIlc` 把 `Channel` 当作完整反馈链路。
5. 最佳ILC输入再次通过同一个 `Channel.Process(..., outputPowerDbm=20.0)` 复测目标工作点。
6. 输出字典同时保留Channel参数与内部PA功率闭环结果，避免把接收噪声功率误解为PA发射功率。

## 9. Channel 特性测量与 DPD 联动

`Channel` 负责施加已配置的 PA 前/后耦合，但不会读取这些配置并自动宣称它们是“测量结果”。独立的 `ChannelAnalyse` 使用逐路探测恢复：

- 主路径和耦合路径的冲激响应；
- 带内幅度平坦度；
- 耦合增益和相位；
- 等效群时延；
- MIMO 频响矩阵条件数。

仿真中可以直接测量两个线性子网络：

```python
from inc.lib.ChannelAnalyse import ChannelAnalyse

channelAnalyzer = ChannelAnalyse(
    parameters={
        "sampleRateHz": 80.0e6,
        "channelBandwidthHz": 20.0e6,
        "width": 0,
    }
)

preMeasurement = channelAnalyzer.Measure(
    channel.ApplyPrePaCoupling,
    chainCount=2,
    stageName="pre-PA",
)
postMeasurement = channelAnalyzer.Measure(
    channel.ApplyPostPaCoupling,
    chainCount=2,
    stageName="post-PA",
)
```

真实硬件必须在对应参考面采集；只有最终端口输出时，一般不能唯一分离 PA 前耦合、PA 非线性和 PA 后耦合。`CouplingAwareDpdGmp` 可以使用上述测量结果修改训练目标并预消除 DAC 侧耦合。

完整测量推导、参数表、实验接线边界、Benchmark 和修改前后性能见 [ChannelAnalyse.md](./ChannelAnalyse.md)。

## 10. 两阶段PA温度测试

热测试与普通“每次都保证目标dBm”的功率扫描不同。`PrepareThermalTest` 首先暂停PA热网络，在参考温度下调用现有闭环一次，随后返回冻结的PA输入；校准过程不产生虚假热量。正式测试只调用不带 `outputPowerDbm` 的 `Process`：

```mermaid
flowchart LR
    raw["任意原始波形"] --> suspend["暂停PA热网络"]
    suspend --> calibrate["一次功率闭环<br/>达到参考温度目标dBm"]
    calibrate --> frozen["冻结PA输入"]
    frozen --> reset["设置起始结温与环境温度"]
    reset --> frame["Process(frozenInput)<br/>开环真实发射"]
    frame --> drift["结温、增益、相位与输出功率自然漂移"]
    drift --> idle["AdvanceThermalIdle<br/>帧间冷却或偏置加热"]
    idle --> frame
```

```python
frozenInput = channel.PrepareThermalTest(
    rawSignal,
    calibrationOutputPowerDbm=22.0,
    initialJunctionTemperatureC=25.0,
    ambientTemperatureC=25.0,
)

for frameIndex in range(100):
    receivedSignal = channel.Process(frozenInput)
    metrics = channel.GetThermalMetrics()
    print(
        frameIndex,
        metrics["junctionTemperatureC"],
        metrics["outputPowerDbm"],
    )
    channel.AdvanceThermalIdle(1.0e-3)
```

三个热接口的作用为：

| 接口 | 是否执行功率闭环 | 是否推进热状态 | 作用 |
|---|---|---|---|
| `PrepareThermalTest(...)` | 是，仅一次 | 否 | 得到参考温度下冻结驱动并设置测试起始温度 |
| `Process(frozenInput)` | 否 | 是 | 真实发射一帧，输出功率允许随温度变化 |
| `AdvanceThermalIdle(idleTimeSec)` | 否 | 是 | RF关闭时按偏置耗散和热网络推进时间 |
| `GetThermalMetrics()` | 否 | 否 | 读取结温、耗散、占空比、有效RF区输出功率和时间；补零不计入输出功率 |

直接调用 `CalibratePaInput` 也会自动暂停并恢复热网络，因此任何功率校准试探都不会改变结温。但温度测试推荐使用 `PrepareThermalTest`，因为它把“校准一次、冻结驱动、复位温度”组合为不易误用的入口。

热源、Foster方程、不同热模型优缺点、全部 `ThermalConfig` 参数和MIMO互热矩阵见 [PaModel.md第13节](./PaModel.md#13-pa电热模型功率占空比与输出漂移)。
