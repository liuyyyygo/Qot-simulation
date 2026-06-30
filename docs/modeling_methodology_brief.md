# 物理损伤感知的低轨卫星光路 QoT 建模流程（简要版，可用于论文正文压缩表述）

本文采用传统最短路加 First-Fit RWA 建立光路，并在光路建立后计算多类物理损伤对单跳和多跳光路 QoT 的影响。建模链路为：链路几何决定自由空间损耗和多普勒频移；光层器件参数决定滤波损耗、OXC 插入损耗和串扰；空间环境决定太阳背景噪声和 EDFA 辐射退化；最终由单跳 OSNR、BER 以及透明多跳 OSNR 累积得到端到端 BER。

## 1. 链路损耗模型

对于星间链路 \(l=(i,j)\)，链路距离为

```math
d_{ij}(t)=\|\mathbf r_j(t)-\mathbf r_i(t)\|.
```

自由空间路径损耗采用 Friis 公式 [1]：

```math
L_{\mathrm{fs},ij}
=20\log_{10}\left(\frac{4\pi d_{ij}}{\lambda}\right).
```

光学天线增益写为 [2]

```math
G_T=\frac{16}{\theta_T^2},\quad
G_R=\left(\frac{\pi D_R}{\lambda}\right)^2.
```

EDFA 前接收功率为

```math
P_{\mathrm{rx,dBm}}
=P_{\mathrm{tx,dBm}}+G_{T,\mathrm{dB}}+G_{R,\mathrm{dB}}
-L_{\mathrm{fs}}-L_D-L_{\mathrm{point}}-L_{\mathrm{OXC}}.
```

决定性因素为链路距离、发散角、接收孔径、指向损耗、OXC 插入损耗和多普勒滤波损耗。

## 2. 多普勒频移和滤波损耗

LEO 卫星相对运动导致多普勒频移 [3], [4]：

```math
\Delta f_D=f_c\frac{v_r}{c}.
```

Yang 等指出，多普勒偏移会影响 WDM 星间光网络中滤波器泄漏和串扰功率惩罚 [4]。本文为网络级仿真采用 super-Gaussian 滤波器近似：

```math
H_D(\Delta f)
=\exp\left[-\left(\frac{|\Delta f|}{B_{\mathrm{eff}}}\right)^{2n}\right],
```

```math
L_D=-10\log_{10}H_D(\Delta f_D).
```

**本文简化：** 该模型不是 Yang 等完整 Lorentz 光谱与拍频功率惩罚模型，而是用于描述多普勒频移导致的等效滤波功率损耗。

## 3. EDFA 辐射退化和 ASE 噪声

空间辐射会引起 EDFA 增益下降和噪声系数上升 [5]-[7]。链路等效剂量取两端卫星平均值：

```math
D_{ij}=\frac{D_i+D_j}{2}.
```

根据低剂量星间光通信 EDFA 辐照实验，5 krad 下输出功率约下降 0.5 dB、NF 约上升 1 dB [5]。因此本文采用

```math
\Delta G_{\mathrm{rad}}=0.10D_{ij},\quad
\Delta NF_{\mathrm{rad}}=0.20D_{ij}.
```

有效增益和噪声系数为

```math
G_{\mathrm{eff,dB}}=G_{0,\mathrm{dB}}-\Delta G_{\mathrm{rad}},
\quad
NF_{\mathrm{eff,dB}}=NF_{0,\mathrm{dB}}+\Delta NF_{\mathrm{rad}}.
```

ASE 噪声采用经典 EDFA 噪声模型 [8], [9]：

```math
P_{\mathrm{ASE}}
=N_{\mathrm{pol}}n_{\mathrm{sp}}h\nu(G_{\mathrm{eff}}-1)B_{\mathrm{ref}},
\quad
n_{\mathrm{sp}}\approx\frac{F_{\mathrm{eff}}}{2}.
```

**本文简化：** 剂量到增益和 NF 的线性关系是低剂量网络级等效标定；若研究具体 EDFA 器件或高剂量场景，应使用 RIA 器件级模型 [6], [7]。

## 4. 天体背景和太阳背景噪声

普通背景光噪声由背景光谱辐亮度、接收孔径、FOV 和滤波器带宽决定 [10]：

```math
P_{\mathrm{sky}}
=\eta L_{\mathrm{sky}}A_R\Omega_{\mathrm{FOV}}\Delta\lambda,
\quad
A_R=\frac{\pi D_R^2}{4}.
```

滤波器频域带宽与光谱带宽关系为

```math
\Delta\lambda=\frac{\lambda_0^2}{c}\Delta f_{\mathrm{filter}}.
```

当太阳进入接收 FOV 时，Liu 等采用 5900 K 黑体辐射近似太阳背景噪声 [11]：

```math
M(\lambda,T_s)
=\frac{2\pi hc^2}{\lambda^5[\exp(hc/(\lambda kT_s))-1]},
```

```math
P_{\mathrm{sun}}
=\frac{\pi^2}{4}D_R^2
\int_{\lambda_1}^{\lambda_2}M(\lambda,T_s)d\lambda.
```

太阳影响判据为 [11]

```math
\theta_{\mathrm{sun}}\le\theta_{\mathrm{FOV}}.
```

本文总背景噪声写为

```math
P_{\mathrm{cel}}
=P_{\mathrm{sky}}+C_{\mathrm{sun}}(\theta_{\mathrm{sun}})P_{\mathrm{sun}}.
```

**本文扩展：** FOV 内太阳受影响判据和黑体积分模型有文献依据 [11]；FOV 外的连续角度耦合函数 \(C_{\mathrm{sun}}\) 是为避免时间序列突变而引入的工程扩展。

## 5. WDM 串扰等效噪声

WDM 星间光网络中的串扰来自 demux/mux 非理想滤波和 OXC 端口泄漏 [4], [13], [14]。对于目标通道 \(k\)，干扰通道 \(m\) 的等效频偏为

```math
\Delta f_{m\rightarrow k}
=(m-k)\Delta f_{\mathrm{ch}}+\Delta f_{D,m}.
```

本文采用等效泄漏系数

```math
\epsilon_{m\rightarrow k}
=10^{-I_{\mathrm{demux}}/10}
\cdot
\exp\left[-\left(\frac{\Delta f_{m\rightarrow k}}{B_{\mathrm{eff}}}\right)^4\right].
```

异波长串扰和同波长 OXC 泄漏方差为

```math
\sigma^2_{\mathrm{inter},k}
=\sum_{m\in\Omega,m\ne k}
(\epsilon_{m\rightarrow k}RP_m)^2,
```

```math
\sigma^2_{\mathrm{intra},k}
=N_{\mathrm{same}}(10^{-I_{\mathrm{OXC}}/10}RP_k)^2.
```

等效串扰噪声功率为

```math
P_{\mathrm{xt},k}
=\frac{\sqrt{\sigma^2_{\mathrm{inter},k}+\sigma^2_{\mathrm{intra},k}}}{R}.
```

**本文简化：** 该模型保留 Yang 等模型中的信道间隔、滤波器带宽、Doppler 频移、器件隔离度和波长占用因素 [4]，但采用链路级等效噪声方差，不是完整节点级 signal-crosstalk beating 功率惩罚模型。

## 6. 单跳 OSNR 和 BER

EDFA 后信号功率为

```math
P_s=P_{\mathrm{rx}}G_{\mathrm{eff}}.
```

单跳总噪声为

```math
P_n=P_{\mathrm{ASE}}+P_{\mathrm{cel}}+P_{\mathrm{xt}}+P_{\mathrm{th}}.
```

因此单跳 OSNR 为

```math
\mathrm{OSNR}_{ij,k}=\frac{P_s}{P_n}.
```

OOK 调制下，基于 OSNR 的 Q 因子和 BER 映射为 [9], [15], [16]

```math
Q\approx
\sqrt{\frac{\mathrm{OSNR}B_{\mathrm{ref}}}{2B_e}},
```

```math
\mathrm{BER}
=\frac{1}{2}\mathrm{erfc}\left(\frac{Q}{\sqrt2}\right).
```

## 7. 多跳透明光路等效 OSNR 和 BER

对于包含 \(H\) 跳的透明光路，多跳 OSNR 采用倒数累积模型 [9], [17]：

```math
\frac{1}{\mathrm{OSNR}_p}
=\sum_{h=1}^{H}
\frac{1}{\mathrm{OSNR}_{l_h,k}}.
```

即

```math
\mathrm{OSNR}_p
=\left(
\sum_{h=1}^{H}
\frac{1}{\mathrm{OSNR}_{l_h,k}}
\right)^{-1}.
```

路径级 BER 继续由

```math
Q_p\approx
\sqrt{\frac{\mathrm{OSNR}_pB_{\mathrm{ref}}}{2B_e}},
\quad
\mathrm{BER}_p
=\frac{1}{2}\mathrm{erfc}\left(\frac{Q_p}{\sqrt2}\right)
```

得到。该模型体现了透明光路中噪声随跳数累积的效应。

## 8. 简化模型边界

严格有文献依据的部分包括自由空间损耗 [1]、多普勒频移 [3], [4]、EDFA ASE 噪声 [8], [9]、太阳背景黑体辐射和 FOV 判据 [11]、WDM 串扰来源 [4], [13], [14]、BER 映射 [15], [16] 以及多跳 OSNR 倒数累积 [9], [17]。

本文为网络级仿真引入的简化包括：SAA 剂量高斯代理场、EDFA 低剂量线性退化、FOV 外太阳噪声连续尾部、super-Gaussian 滤波损耗和链路级串扰等效噪声。这些简化应在论文中明确说明，避免被误解为完全复现器件级或节点级物理模型。

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

