# 物理损伤感知的低轨卫星光路 QoT 建模方法（详细版，可用于论文方法部分）

本文面向光交换架构下的低轨卫星光网络，研究传统最短路加 First-Fit 路由与波长分配策略在多类物理损伤共同作用下的传输性能。不同于 QoT 感知路由算法，本文首先按照网络层可达性和波长连续性建立光路，然后对已建立光路进行物理层 QoT 评估。建模流程为：首先根据卫星轨道几何计算星间链路距离、相对径向速度和太阳入射几何；其次计算链路损耗、EDFA 辐射退化、背景光噪声、ASE 噪声、WDM 串扰噪声和热噪声；最后由单跳 OSNR 映射得到单跳 BER，并通过透明多跳 OSNR 累积得到端到端光路 BER。

为保证模型具有文献依据，本文将严格文献支撑的物理模型与为适配大规模网络仿真而引入的等效简化模型分开描述。文中标注“本文简化”或“本文扩展”的部分不声明为原文献的直接公式，而是基于已有理论的网络级等效实现。

## 1. 星座几何、星间链路与光路定义

设低轨卫星星座由 \(N_p\) 个轨道面构成，每个轨道面包含 \(N_s\) 颗卫星。本文采用 circular orbit 模型描述卫星运动，不引入 SGP4 或星历接口。对于任意卫星 \(i\)，其在时刻 \(t\) 的位置和速度分别记为 \(\mathbf r_i(t)\) 和 \(\mathbf v_i(t)\)。对于星间激光链路 \(l=(i,j)\)，链路距离为

```math
d_{ij}(t)=\|\mathbf r_j(t)-\mathbf r_i(t)\|.
```

链路方向单位向量为

```math
\mathbf u_{ij}(t)=\frac{\mathbf r_j(t)-\mathbf r_i(t)}{d_{ij}(t)}.
```

相对径向速度定义为

```math
v_{r,ij}(t)=\left[\mathbf v_j(t)-\mathbf v_i(t)\right]\cdot\mathbf u_{ij}(t).
```

上述几何量用于后续自由空间路径损耗、多普勒频移和太阳背景噪声角度判据的计算。本文采用传统最短路加 First-Fit 建立光路。对于业务请求 \(r=(s,d,b)\)，其中 \(s\) 为源卫星，\(d\) 为宿卫星，\(b\) 为业务速率，首先在星间拓扑上求最短路径，然后在满足波长连续性的条件下选择编号最小的可用波长。默认仿真模式不在路由阶段根据 OSNR 或 BER 拒绝路径，而是在光路建立后计算 QoT 指标。

## 2. 链路损耗建模

### 2.1 自由空间路径损耗

星间激光链路的自由空间路径损耗采用 Friis 传输公式 [1]。设通信波长为 \(\lambda\)，星间距离为 \(d_{ij}\)，则自由空间路径损耗为

```math
L_{\mathrm{fs},ij}
=20\log_{10}\left(\frac{4\pi d_{ij}}{\lambda}\right).
```

该公式严格来源于自由空间传播理论 [1]，反映了链路距离和通信波长对接收功率的决定作用。由于低轨星座中轨道间链路长度随时间变化，\(L_{\mathrm{fs},ij}\) 也是随轨道位置变化的时变损耗项。

### 2.2 光学天线增益

发射端光学增益主要由激光束发散角决定，接收端光学增益主要由接收孔径决定。本文采用光链路预算中常用的光学天线增益形式 [2]：

```math
G_T=\frac{16}{\theta_T^2},
```

```math
G_R=\left(\frac{\pi D_R}{\lambda}\right)^2,
```

其中，\(\theta_T\) 为发射半角发散角，\(D_R\) 为接收端孔径直径。dB 形式为

```math
G_{T,\mathrm{dB}}=10\log_{10}G_T,\quad
G_{R,\mathrm{dB}}=10\log_{10}G_R.
```

较小的发散角可以提高发射增益，较大的接收孔径可以提高接收增益；但接收孔径同时会放大太阳背景噪声接收功率，这一点在背景噪声模型中进一步体现。

### 2.3 固定指向损耗和 OXC 插入损耗

星间激光链路还包含指向误差引入的固定损耗和光交换节点插入损耗。本文将其建模为每跳固定 dB 损耗：

```math
L_{\mathrm{fix}}=L_{\mathrm{point}}+L_{\mathrm{OXC}}.
```

其中，\(L_{\mathrm{point}}\) 为指向损耗，\(L_{\mathrm{OXC}}\) 为光交叉连接或光交换节点的插入损耗。该部分属于工程链路预算项，主要用于反映 ATP 指向误差、光学器件耦合损耗和 OXC 透传损耗。

### 2.4 多普勒频移和滤波损耗

低轨卫星高速运动会导致星间链路出现多普勒频移。设光载频为 \(f_c\)，径向相对速度为 \(v_{r,ij}\)，则多普勒频移为 [3], [4]

```math
\Delta f_{D,ij}=f_c\frac{v_{r,ij}}{c}.
```

Yang 等指出，LEO 星间 WDM 光网络中，不同输入 ISL 的多普勒波长偏移会改变波长路由节点中的串扰功率惩罚，并使系统惩罚随星座运动周期性波动 [4]。同时，信号频偏也会导致接收光滤波器失配，从而降低目标信号通过滤波器的功率 [4]。

**本文简化。** 为适配网络级仿真，本文不展开 Yang 等文中的 Lorentz 光谱与拍频噪声功率惩罚推导，而采用 super-Gaussian 光滤波器近似描述多普勒频偏导致的接收功率衰减。设光滤波器 3 dB 带宽为 \(B_{3\mathrm{dB}}\)，滤波器阶数为 \(n\)，则

```math
H_D(\Delta f)
=\exp\left[-\left(\frac{|\Delta f|}{B_{\mathrm{eff}}}\right)^{2n}\right],
```

```math
B_{\mathrm{eff}}
=\frac{B_{3\mathrm{dB}}}{(\ln2)^{1/(2n)}}.
```

多普勒滤波损耗定义为

```math
L_{D,ij}
=-10\log_{10}H_D(\Delta f_{D,ij}).
```

该模型保留了多普勒频移、滤波器带宽和滤波器形状对信号功率的决定作用，但属于本文为简化大规模仿真而采用的等效滤波损耗模型。

### 2.5 单跳接收信号功率

综合自由空间损耗、固定损耗和多普勒滤波损耗，EDFA 前接收功率为

```math
P_{\mathrm{rx,dBm},ij}
=P_{\mathrm{tx,dBm}}
+G_{T,\mathrm{dB}}
+G_{R,\mathrm{dB}}
-L_{\mathrm{fs},ij}
-L_{\mathrm{point}}
-L_{\mathrm{OXC}}
-L_{D,ij}.
```

将 dBm 转换为线性功率，并考虑光学接收效率 \(\eta\)，得到

```math
P_{\mathrm{rx},ij}
=10^{(P_{\mathrm{rx,dBm},ij}-30)/10}\eta.
```

该功率作为后续 EDFA 放大、串扰噪声和 OSNR 计算的基础。

## 3. EDFA 辐射退化与 ASE 噪声建模

空间辐射会引起掺铒光纤中的 radiation-induced attenuation，进而导致 EDFA 增益下降和噪声系数上升 [5], [6], [7]。Giles 和 Desurvire 的 EDFA 建模工作表明，EDFA 的增益、噪声系数和 ASE 噪声共同决定放大器输出信号质量 [8]。

### 3.1 累计剂量与等效链路剂量

设卫星 \(i\) 和卫星 \(j\) 的任务累计辐射剂量分别为 \(D_i\) 和 \(D_j\)，本文将链路两端平均剂量作为该跳链路的等效 EDFA 辐射剂量：

```math
D_{ij}=\frac{D_i+D_j}{2}.
```

**本文简化。** 当前仿真中的 SAA 剂量空间分布采用二维高斯代理模型，用于产生不同卫星之间的剂量差异。该高斯场不应表述为严格空间辐射环境模型。若论文需要更严格剂量输入，应由 AE9/AP9、SPENVIS 或具体屏蔽厚度下的 TID 分析给出 \(D_i\)。

### 3.2 EDFA 增益下降和噪声系数上升

低剂量轨道下，Li 等针对星间光通信 EDFA 的电子辐照实验表明，在 5 krad 剂量下，EDFA 输出功率约下降 0.5 dB，噪声系数约上升 1 dB [5]。据此，本文采用低剂量线性等效模型：

```math
\Delta G_{\mathrm{rad}}(D_{ij})=k_GD_{ij},
```

```math
\Delta NF_{\mathrm{rad}}(D_{ij})=k_{NF}D_{ij},
```

其中

```math
k_G=\frac{0.5}{5}=0.10~\mathrm{dB/krad},
```

```math
k_{NF}=\frac{1.0}{5}=0.20~\mathrm{dB/krad}.
```

有效 EDFA 增益和有效噪声系数为

```math
G_{\mathrm{eff,dB}}
=G_{0,\mathrm{dB}}-\Delta G_{\mathrm{rad}}(D_{ij}),
```

```math
NF_{\mathrm{eff,dB}}
=NF_{0,\mathrm{dB}}+\Delta NF_{\mathrm{rad}}(D_{ij}).
```

**本文简化。** 上述线性模型是根据低剂量实验点进行的网络级等效标定，适合用于低剂量轨道场景下的 QoT 敏感性分析；它不替代器件级 EDFA 速率方程、泵浦传播方程或 RIA 光谱建模。若研究高剂量或具体器件结构，应采用 Facchini 等和 Ladaci 等基于 RIA 的 EDFA 器件级模型 [6], [7]。

### 3.3 ASE 噪声功率

将有效增益和有效噪声系数转换为线性形式：

```math
G_{\mathrm{eff}}=10^{G_{\mathrm{eff,dB}}/10},
```

```math
F_{\mathrm{eff}}=10^{NF_{\mathrm{eff,dB}}/10}.
```

在高增益近似下，自发辐射因子为 [8], [9]

```math
n_{\mathrm{sp}}\approx\frac{F_{\mathrm{eff}}}{2}.
```

参考光带宽 \(B_{\mathrm{ref}}\) 内的双偏振 ASE 噪声功率为 [8], [9]

```math
P_{\mathrm{ASE},ij}
=N_{\mathrm{pol}}\cdot n_{\mathrm{sp}}\cdot h\nu
\cdot\left(G_{\mathrm{eff}}-1\right)\cdot B_{\mathrm{ref}},
```

其中，\(N_{\mathrm{pol}}=2\)，\(h\) 为普朗克常数，\(\nu=c/\lambda\)。因此，辐射导致的 EDFA 影响通过两条路径进入 QoT：增益下降降低 EDFA 后信号功率，噪声系数上升增大 ASE 噪声。本文不将 NF 上升作为链路损耗直接相加，而是通过 \(P_{\mathrm{ASE}}\) 进入总噪声。

EDFA 后信号功率为

```math
P_{s,ij}=P_{\mathrm{rx},ij}G_{\mathrm{eff}}.
```

## 4. 天体背景与太阳背景噪声建模

自由空间光通信中的背景照明会提高接收端噪声底，从而降低 SNR/OSNR [10]。对于低轨星间激光链路，太阳进入接收终端 FOV 时可能引起显著太阳背景噪声，降低接收灵敏度并影响捕获跟踪性能 [11], [12]。

### 4.1 普通天体背景噪声

普通天空或深空背景噪声可由背景光谱辐亮度、接收孔径、接收视场和滤波器光谱带宽共同确定 [10]。设背景光谱辐亮度为 \(L_{\mathrm{sky}}\)，接收孔径面积为

```math
A_R=\frac{\pi D_R^2}{4},
```

接收视场立体角为 \(\Omega_{\mathrm{FOV}}\)，光谱带宽为 \(\Delta\lambda\)，光学效率为 \(\eta\)，则

```math
P_{\mathrm{sky}}
=\eta L_{\mathrm{sky}}A_R\Omega_{\mathrm{FOV}}\Delta\lambda.
```

光谱带宽由光滤波器频域带宽换算：

```math
\Delta\lambda
=\frac{\lambda_0^2}{c}\Delta f_{\mathrm{filter}}.
```

该公式体现了背景噪声与接收孔径、视场和光滤波器带宽之间的比例关系。

### 4.2 太阳直射背景噪声

Liu 等在 LEO 巨型星座太阳背景噪声分析中将太阳光谱近似为 \(T_s\approx5900\,\mathrm K\) 的黑体辐射，并用普朗克公式表示太阳光谱辐射出射度 [11]：

```math
M(\lambda,T_s)
=\frac{2\pi hc^2}
{\lambda^5\left[\exp\left(hc/(\lambda kT_s)\right)-1\right]}.
```

当太阳进入接收端 FOV 时，太阳背景噪声功率可写为 [11]

```math
P_{\mathrm{sun}}
=\frac{\pi^2}{4}D_R^2
\cdot\int_{\lambda_1}^{\lambda_2}M(\lambda,T_s)d\lambda.
```

若光滤波器带宽较窄，则可近似为

```math
P_{\mathrm{sun}}
\approx
\frac{\pi^2}{4}D_R^2M(\lambda_0,T_s)\Delta\lambda.
```

Liu 等给出的示例参数包括 \(D_R=10\) mm、\(\lambda_0=1550\) nm、波段宽度约 1 nm，并报告太阳背景噪声可达到与通信信号同量级的水平 [11]。

### 4.3 太阳进入 FOV 判据与本文角度扩展

设接收光轴方向与太阳方向之间的夹角为 \(\theta_{\mathrm{sun}}\)，若

```math
\theta_{\mathrm{sun}}\le \theta_{\mathrm{FOV}},
```

则认为太阳进入接收端视场，链路受到太阳背景噪声影响 [11]。

**本文扩展。** 严格的文献判据是二值 FOV 判据。为避免时间序列中太阳背景噪声从 0 到强噪声的突变，本文在 FOV 外加入经验型连续角度耦合项：

```math
C_{\mathrm{sun}}(\theta_{\mathrm{sun}})
=
\begin{cases}
1, & \theta_{\mathrm{sun}}\le\theta_{\mathrm{FOV}},\\
\exp\left[-\left(\frac{\theta_{\mathrm{sun}}-\theta_{\mathrm{FOV}}}{\theta_0}\right)^m\right],
& \theta_{\mathrm{sun}}>\theta_{\mathrm{FOV}}.
\end{cases}
```

因此天体背景总噪声为

```math
P_{\mathrm{cel},ij}
=P_{\mathrm{sky}}
+C_{\mathrm{sun}}(\theta_{\mathrm{sun},ij})P_{\mathrm{sun}}.
```

其中 \(C_{\mathrm{sun}}\) 的指数尾部是本文为连续化和灵敏度分析引入的工程扩展，不应表述为 Liu 等原文模型。

## 5. WDM 串扰等效噪声建模

光交换架构下，WDM 星间光网络中的串扰主要来自波长路由节点中 demux/mux 滤波器的非理想隔离和光开关/OXC 端口泄漏 [4], [13], [14]。Yang 等针对 WDM 光卫星网络指出，Doppler wavelength shift 会改变不同输入 ISL 中相同标称波长之间的频率关系，从而引起串扰功率惩罚波动 [4]。

### 5.1 异波长泄漏串扰

对于目标通道 \(k\)，设当前链路上已占用波长集合为 \(\Omega\)。干扰通道 \(m\) 相对于目标通道的等效频偏为

```math
\Delta f_{m\rightarrow k}
=(m-k)\Delta f_{\mathrm{ch}}+\Delta f_{D,m},
```

其中 \(\Delta f_{\mathrm{ch}}\) 为信道间隔，\(\Delta f_{D,m}\) 为干扰通道对应链路的多普勒频移。

**本文简化。** Yang 等完整模型从电场叠加出发，计算 signal-crosstalk beating noise 及其功率惩罚 [4]。本文为适配大规模 RWA 仿真，将滤波器泄漏建模为等效串扰噪声方差。采用四阶 super-Gaussian 旁瓣透过率时，异波长泄漏系数为

```math
\epsilon_{m\rightarrow k}
=10^{-I_{\mathrm{demux}}/10}
\cdot
\exp\left[-\left(\frac{\Delta f_{m\rightarrow k}}{B_{\mathrm{eff}}}\right)^4\right],
```

其中 \(I_{\mathrm{demux}}\) 为 demux/mux 隔离度。异波长串扰方差为

```math
\sigma^2_{\mathrm{inter},k}
=\sum_{m\in\Omega,m\ne k}
\left(\epsilon_{m\rightarrow k}RP_m\right)^2,
```

其中 \(R\) 为光电探测响应度，\(P_m\) 为干扰通道接收功率。

### 5.2 同波长 OXC 泄漏串扰

同波长串扰来自其他输入端口的相同标称波长通过 OXC 或 optical switch 泄漏到目标输出端 [4], [13], [14]。设同波长干扰端口数为 \(N_{\mathrm{same}}\)，OXC 隔离度为 \(I_{\mathrm{OXC}}\)，则

```math
XT_{\mathrm{OXC}}=10^{-I_{\mathrm{OXC}}/10},
```

```math
\sigma^2_{\mathrm{intra},k}
=N_{\mathrm{same}}\left(XT_{\mathrm{OXC}}RP_k\right)^2.
```

串扰总方差为

```math
\sigma^2_{\mathrm{xt},k}
=\sigma^2_{\mathrm{inter},k}
+\sigma^2_{\mathrm{intra},k}.
```

为与 ASE、背景光和热噪声共同进入 OSNR，本文将串扰方差转换为等效噪声功率：

```math
P_{\mathrm{xt},k}
=\frac{\sqrt{\sigma^2_{\mathrm{xt},k}}}{R}.
```

该模型保留了 WDM 串扰的关键决定因素：信道间隔、滤波器带宽、demux 隔离度、OXC 隔离度、波长占用状态和 Doppler 频移。需要强调的是，当前实现属于链路级等效串扰噪声模型，不是 Yang 等的完整节点级 OXC 拍频功率惩罚模型 [4]。

## 6. 热噪声

接收机热噪声可由等效噪声功率直接给出：

```math
P_{\mathrm{th}}
=10^{(P_{\mathrm{th,dBm}}-30)/10}.
```

若从物理热噪声角度表示，也可写为

```math
P_{\mathrm{th}}=k_BT B_eF_{\mathrm{rx}},
```

其中 \(T\) 为接收机温度，\(B_e\) 为电接收带宽，\(F_{\mathrm{rx}}\) 为接收机噪声系数。当前仿真将 \(P_{\mathrm{th,dBm}}\) 作为配置参数输入。

## 7. 单跳 OSNR 建模

对于链路 \(l=(i,j)\) 上的波长通道 \(k\)，单跳总噪声为

```math
P_{n,ij,k}
=P_{\mathrm{ASE},ij}
+P_{\mathrm{cel},ij}
+P_{\mathrm{xt},ij,k}
+P_{\mathrm{th}}.
```

单跳 OSNR 为

```math
\mathrm{OSNR}_{ij,k}
=\frac{P_{s,ij}}{P_{n,ij,k}}.
```

dB 形式为

```math
\mathrm{OSNR}_{ij,k,\mathrm{dB}}
=10\log_{10}\left(\mathrm{OSNR}_{ij,k}\right).
```

由于 \(P_{\mathrm{xt},ij,k}\) 与波长占用相关，单跳 OSNR 是波长相关的。由于 \(P_{\mathrm{cel},ij}\) 与太阳夹角相关，单跳 OSNR 也是时变的。

## 8. BER 映射

本文采用 OOK 调制下基于 OSNR 的 Gaussian 近似 BER 映射。Marcuse、Humblet 和 Azizoglu 等对含光放大器的光通信系统 BER 建模进行了经典分析 [15], [16]；Agrawal 也给出了 OSNR、Q 因子和 BER 之间的常用关系 [9]。设 OSNR 参考光带宽为 \(B_{\mathrm{ref}}\)，接收电带宽为 \(B_e\)，则

```math
Q\approx
\sqrt{\frac{\mathrm{OSNR}\,B_{\mathrm{ref}}}{2B_e}}.
```

BER 为

```math
\mathrm{BER}
=\frac{1}{2}\mathrm{erfc}
\left(\frac{Q}{\sqrt{2}}\right).
```

若考虑有限消光比 \(ER_{\mathrm{dB}}\)，可令

```math
r_{\mathrm{ex}}=10^{-ER_{\mathrm{dB}}/10},
```

并采用修正 Q 因子

```math
Q_{\mathrm{ex}}
=\sqrt{2\mathrm{OSNR}\frac{B_{\mathrm{ref}}}{B_e}}
\frac{1-r_{\mathrm{ex}}}{1+\sqrt{r_{\mathrm{ex}}}}.
```

当前仿真默认采用不含消光比修正的 OSNR-Q 映射。图中可同时标注 \(10^{-7}\) 的 QoT 劣化观察门限和 \(2\times10^{-3}\) 的 HD-FEC 门限。

## 9. 透明多跳光路等效 OSNR 与 BER

对于包含 \(H\) 条链路的透明光路

```math
p=\{l_1,l_2,\ldots,l_H\},
```

由于光信号在中间节点不进行电再生，各跳噪声会沿透明路径累积。多跳 OSNR 可采用倒数累积模型 [9], [17]：

```math
\frac{1}{\mathrm{OSNR}_p}
=\sum_{h=1}^{H}
\frac{1}{\mathrm{OSNR}_{l_h,k}}.
```

因此

```math
\mathrm{OSNR}_p
=\left(
\sum_{h=1}^{H}
\frac{1}{\mathrm{OSNR}_{l_h,k}}
\right)^{-1}.
```

路径级 Q 因子和 BER 为

```math
Q_p\approx
\sqrt{\frac{\mathrm{OSNR}_pB_{\mathrm{ref}}}{2B_e}},
```

```math
\mathrm{BER}_p
=\frac{1}{2}\mathrm{erfc}
\left(\frac{Q_p}{\sqrt2}\right).
```

该模型反映了透明多跳光路的基本特性：即使每一跳都具备可接受的 OSNR，随着跳数增加，端到端 OSNR 仍会下降，BER 可能显著恶化。因此，在传统最短路加 First-Fit RWA 下，网络层可建立的光路并不一定满足物理层 QoT 要求。

## 10. 本文模型边界

1. 自由空间损耗、光学天线增益、多普勒频移、EDFA ASE 噪声、太阳背景黑体辐射、WDM 串扰来源、OSNR-BER 映射和多跳 OSNR 累积均具有明确文献依据 [1]-[17]。

2. EDFA 辐射退化的线性参数由低剂量实验点标定 [5]，适合作为低剂量网络级 QoT 模型；若研究高剂量或具体器件，应采用 RIA 器件级模型 [6], [7]。

3. SAA 剂量场是本文仿真的空间代理模型，不是严格辐射环境模型。论文中应表述为“简化 SAA 暴露代理”或将剂量视为外部输入。

4. 太阳背景噪声的 FOV 内判据和黑体积分具有文献依据 [11]；FOV 外指数角度尾部是本文为连续化 QoT 时间序列引入的工程扩展。

5. WDM 串扰模型保留了 Yang 等模型的关键物理因素 [4]，但当前实现为链路级等效噪声方差模型，不是完整节点级拍频功率惩罚模型。

## 参考文献

[1] H. T. Friis, "A Note on a Simple Transmission Formula," Proceedings of the IRE, vol. 34, no. 5, pp. 254-256, 1946.

[2] J. Liang, A. U. Chaudhry, E. Erdogan, and H. Yanikomeroglu, "Link Budget Analysis for Free-Space Optical Satellite Networks," IEEE WoWMoM, pp. 471-476, 2022.

[3] W. Boumalek, S. Aris, S. T. Goh, S. A. Zekavat, and M. Benslama, "LEO Satellite Constellations Configuration Based on the Doppler Effect in Laser Intersatellite Links," International Journal of Satellite Communications and Networking, vol. 42, no. 2, 2024.

[4] Q. Yang, L. Tan, and J. Ma, "Analysis of Crosstalk in Optical Satellite Networks With Wavelength Division Multiplexing Architectures," Journal of Lightwave Technology, vol. 28, no. 6, pp. 931-938, 2010.

[5] M. Li, J. Ma, L. Tan, Y. Zhou, S. Yu, et al., "Radiation effect on EDFA for inter-satellite optical communication on low dose orbits," Proc. SPIE, vol. 7134, 713408, 2008.

[6] A. Facchini, A. Morana, L. Mescia, C. Campanella, M. M. K. Shuvo, T. Robin, E. Marin, Y. Ouerdane, A. Boukenter, and S. Girard, "Experimental-Simulation Analysis of a Radiation Tolerant Erbium-Doped Fiber Amplifier for Space Applications," Applied Sciences, vol. 13, no. 20, 11589, 2023.

[7] A. Ladaci, S. Girard, L. Mescia, et al., "Optimized Radiation-Hardened Erbium Doped Fiber Amplifiers for Long Space Missions," Journal of Applied Physics, vol. 121, 163104, 2017.

[8] C. R. Giles and E. Desurvire, "Modeling Erbium-Doped Fiber Amplifiers," Journal of Lightwave Technology, vol. 9, no. 2, pp. 271-283, 1991.

[9] G. P. Agrawal, Fiber-Optic Communication Systems, 5th ed., Wiley, 2021.

[10] W. R. Leeb, "Degradation of Signal to Noise Ratio in Optical Free Space Data Links Due to Background Illumination," Applied Optics, vol. 28, no. 16, pp. 3443-3449, 1989.

[11] Y. Liu, J. Bai, M. Sheng, et al., "Impact Analysis of Solar Background Noise on LEO Mega-Constellations," IEEE International Conference on Communications, 2025.

[12] Q. Yang, L. Tan, Q. Han, J. Ma, and W. Xie, "Celestial background noise analysis for laser intersatellite links," Applied Optics, 2008.

[13] E. L. Goldstein, L. Eskildsen, and A. F. Elrefaie, "Performance Implications of Component Crosstalk in Transparent Lightwave Networks," IEEE Photonics Technology Letters, vol. 6, no. 5, pp. 657-660, 1994.

[14] Y. Shen, K. Lu, and W. Gu, "Coherent and Incoherent Crosstalk in WDM Optical Networks," Journal of Lightwave Technology, vol. 17, no. 5, pp. 759-764, 1999.

[15] D. Marcuse, "Derivation of Analytical Expressions for the Bit-Error Probability in Lightwave Systems with Optical Amplifiers," Journal of Lightwave Technology, vol. 8, no. 12, pp. 1816-1823, 1990.

[16] P. A. Humblet and M. Azizoglu, "On the Bit Error Rate of Lightwave Systems with Optical Amplifiers," Journal of Lightwave Technology, vol. 9, no. 11, pp. 1576-1582, 1991.

[17] R. Ramaswami, K. N. Sivarajan, and G. H. Sasaki, Optical Networks: A Practical Perspective, 3rd ed., Morgan Kaufmann, 2010.

