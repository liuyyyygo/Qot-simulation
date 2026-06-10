# 参考文献汇总：星间激光链路轨道相关性物理损伤建模

## 1. 自由空间损耗 (Free Space Loss)

- **H.T. Friis**, "A Note on a Simple Transmission Formula," *Proc. IRE*, 34(5), 254-256, 1946. — 原始Friis传输方程
- **Y. Suh & Y. Ko**, "Comprehensive Optical Inter-Satellite Communication Model for LEO Constellations," *J-KICS*, 50(12), 1872-1884, 2025. — LEO ISL链路预算，包含六种指向误差模型
- **J. Liang, A.U. Chaudhry, E. Erdogan, H. Yanikomeroglu**, "Link Budget Analysis for Free-Space Optical Satellite Networks," *IEEE WoWMoM*, 2022, pp.471-476.
- **I.P. Vieira, T.C. Pita, D. Mello**, "Link-length Analysis for First-neighbor Optical ISLs in Ultra-dense LEO Constellations," *SBrT*, 2022. — ISL距离计算与FSL

## 2. 多普勒频移与滤波器功率惩罚

- **W. Boumalek, S. Aris, S.T. Goh, S.A. Zekavat, M. Benslama**, "LEO Satellite Constellations Configuration Based on the Doppler Effect in Laser Intersatellite Links," *Int. J. Satell. Commun. Netw.*, 42(2), 2024. — LEO星座参数对多普勒波长漂移的系统分析
- **Q. Yang, L. Tan, J. Ma**, "Analysis of Crosstalk in Optical Satellite Networks With Wavelength Division Multiplexing Architectures," *J. Lightwave Technol.*, 28(6), 931-938, 2010. — WDM星间链路多普勒串扰联合分析（**核心参考文献**）
- **A.D. Barman, A. Halder**, "Performance Analysis of Doppler Shift Induced Optical Filtering and Inband Crosstalk Penalties in Intersatellite Link Optical Switching Nodes," *CODEC*, 2012, pp.1-4. — 多普勒频移引起的滤波损失与带内串扰
- **M. Toyoshima et al.**, "Maximum Doppler shift in inter-orbit laser communication," *Proc. SPIE*, 2005.

## 3. 天体背景光噪声

- **W.R. Leeb**, "Degradation of Signal to Noise Ratio in Optical Free Space Data Links Due to Background Illumination," *Applied Optics*, 28(16), 3443-3449, 1989. — 太阳/月球/行星背景光SNR退化经典定量分析
- **S.-P. Chen**, "Performance Analysis of Near-Earth, Lunar and Interplanetary Optical Communication Links," *Opt. Quant. Electron.*, 54:562, 2022. — 近地光通信链路背景噪声功率模型
- **N.K. Lyras et al.**, IEEE Aerospace & Electronic Systems, Nov. 2019. — 背景噪声功率一般模型
- **C.R. Kitchin**, *Astrophysical Techniques*, 6th ed., CRC Press, 2014. — 天体光谱辐亮度参考数据

## 4. WDM串扰

- **Q. Yang, L. Tan, J. Ma**, "Analysis of Crosstalk in Optical Satellite Networks With Wavelength Division Multiplexing Architectures," *J. Lightwave Technol.*, 28(6), 931-938, 2010. — 星载WDM OXC串扰模型（**核心参考文献**）
- **E.L. Goldstein, L. Eskildsen, A.F. Elrefaie**, "Performance Implications of Component Crosstalk in Transparent Lightwave Networks," *IEEE PTL*, 6(5), 657-660, 1994. — 透明光网络串扰经典分析
- **J.Y. Zhou et al.**, "Crosstalk in Multiwavelength Optical Cross-Connect Networks," *JLT*, 14(6), 1423-1435, 1996. — 多波长OXC串扰基础理论
- **ITU-T Recommendation G.692**, "Optical interfaces for multichannel systems with optical amplifiers."
- **R. Ramaswami, K.N. Sivarajan, G.H. Sasaki**, *Optical Networks: A Practical Perspective*, 3rd ed., Morgan Kaufmann, 2010, Ch.8.

## 5. 空间辐射对EDFA的影响

- **A. Facchini, A. Morana, L. Mescia, C. Campanella, et al.**, "Experimental-Simulation Analysis of a Radiation Tolerant Erbium-Doped Fiber Amplifier for Space Applications," *Applied Sciences*, 13(20), 11589, 2023. — 辐射耐受EDFA实验与仿真分析（**核心参考文献**，3kGy测试，含光漂白效应）
- **A. Ladaci, S. Girard, L. Mescia, et al.**, "Optimized Radiation-Hardened Erbium Doped Fiber Amplifiers for Long Space Missions," *J. Applied Physics*, 121(16), 163104, 2017. — 长周期空间任务辐射加固EDFA优化
- **M. Aubry et al.**, "Combined Temperature and Radiation Effects on the Gain of Er- and Er-Yb-Doped Fiber Amplifiers," *IEEE TNS*, 68(5), 964-971, 2021. — 温度-辐射耦合效应
- **S. Girard et al.**, "Radiation Effects on Silica-Based Optical Fibers: Recent Advances and Future Challenges," *J. Optics*, 20, 093001, 2018. — 光纤辐射效应综述
- **C.R. Giles, E. Desurvire**, "Modeling Erbium-Doped Fiber Amplifiers," *JLT*, 9(2), 271-283, 1991. — EDFA基础模型
- **AE9/AP9/SPM**, NASA标准辐射带模型 — SAA辐射环境建模
- **M.N. Ott**, "Radiation Effects Data on Commercially Available Optical Fiber: Database Summary," *IEEE REDW*, 2002.

## 6. OSNR累积模型（多跳透传光路）

- **G.P. Agrawal**, *Fiber-Optic Communication Systems*, 5th ed., Wiley, 2021, Ch.7. — OSNR累积与噪声标准模型
- **R. Ramaswami et al.**, *Optical Networks*, 3rd ed., Ch.5. — 1/OSNR_total = Σ(1/OSNR_i) 标准累积公式
- **F. Francis, R. Manivasakan**, "A Performance Limit Estimation Framework for Multihop Repeated/Regenerated Optical Links," *IEEE Access*, 10, 70016-70031, 2022. — 多跳光链路性能评估框架

## 7. OSNR → BER映射

- **D. Marcuse**, "Derivation of Analytical Expressions for the Bit-Error Probability in Lightwave Systems with Optical Amplifiers," *JLT*, 8(12), 1816-1823, 1990. — OOK光放大系统BER精确解
- **P.A. Humblet, M. Azizoglu**, "On the Bit Error Rate of Lightwave Systems with Optical Amplifiers," *JLT*, 9(11), 1576-1582, 1991. — 高斯近似的验证
- **G.P. Agrawal**, *Fiber-Optic Communication Systems*, 5th ed., Wiley, 2021, Ch.7. — BER = 0.5·erfc(Q/√2) 标准映射
- **N.S. Bergano et al.**, "Margin measurements in optical amplifier systems," *IEEE PTL*, 5(3), 304-306, 1993. — OSNR-Q因子关系
- **ITU-T G-Series Supplement 39**, "Optical system design and engineering considerations," 2016. — FEC门限标准

## 8. RWA与QoT约束

- **H. Zang, J.P. Jue, B. Mukherjee**, "A Review of Routing and Wavelength Assignment Approaches," *IEEE Network*, 14(1), 2000. — RWA综述
- **E.L. Goldstein et al.**, "Scaling limitations in transparent optical networks due to crosstalk," *OFC*, 1995. — 串扰对透明光网络可扩展性的限制

---

## 论文中引用标注问题的修正

论文正文中：
- 第2.3节正文引用"Yang et al. [4]"分析WDM LISL中多普勒波长漂移，但参考文献列表[4]实际为**B.S. Robinson**等人的月球激光通信演示论文
- 第2.3节正文引用"Yang et al. [5]"计算天体背景噪声，但参考文献列表[5]实际为**Y. Arimoto**等人的ETS-VI实验论文

建议补充以下正确引用：
- 多普勒分析引用 **Q. Yang, L. Tan, J. Ma, JLT 2010**（见上述第2节）
- 天体背景噪声引用 **Q. Yang 等**相关论文 或 改用 **W.R. Leeb 1989** 和 **S.-P. Chen 2022**（见上述第3节）
