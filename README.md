# QDD 关节无力矩传感器力矩估计

单关节仿真代码。目标是不用力矩传感器，只用电机电流和编码器来估计关节输出力矩。

Python 标准库实现，不依赖 numpy / matplotlib。

## 运行

    python src/simulate_stdlib.py

结果输出到 `docs/figures/`：`metrics_stdlib.txt` 是指标，`trace.csv` 是轨迹数据。

其他脚本：

    python src/regime_study.py      不同辨识协议的对比（4 个任务）
    python src/error_budget.py      齿槽转矩与温漂对精度的影响
    python src/temp_comp.py         温漂补偿
    python src/mismatch_study.py    真实模型与辨识模型不一致时的表现
    python src/design_study.py      关节参数与噪声的敏感性
    python src/identify_compare.py  最小二乘与 EKF 的对比
    python src/make_figures.py      生成 SVG 图

`src/simulate.py` 是 numpy + matplotlib 版本，本来用来出图，但这台机器上没装这两个库，
所以没有跑过，保留备用。

## 文件说明

    src/joint_model.py        单关节动力学 + Stribeck 摩擦（当作"真实系统"）
    src/momentum_observer.py  广义动量观测器
    src/friction_rls.py       RLS 在线摩擦辨识
    src/simulate_stdlib.py    主程序，零依赖
    docs/figures/             图与指标输出
    experiments.md            实验记录（做过的尝试、失败的假设）

## 结果

接触段力矩估计误差（RMSE，N·m，任务 T1）：

    纯电流 τ = Kt·iq          0.174
    残差直接用（不扣摩擦）     0.186
    观测器 + 摩擦补偿          0.030
    用真实摩擦模型补偿         0.066

比纯电流基线降低 82.6%。稳态下观测器残差与真实摩擦的偏差小于 1%。

最后一行用真实摩擦反而更差，这一点在 `experiments.md` 里有排查记录：
把测速噪声设为 0 结果几乎不变，说明不是噪声问题；提高观测器带宽能缩小
差距但不归零。目前的解释是观测器残差本身是被滤波过的摩擦，辨识和补偿
走同一个通道时系统误差会抵消。

## 目前的问题

- 结果全部来自仿真，没有硬件验证
- Stribeck 参数不可辨识：τs 和 vs 在有限速度激励下无法分开确定，参数误差
  40~90%。但力矩曲线的误差只有 0.010 N·m，所以对补偿影响不大
- 温漂只能补偿一部分（保持常数而漂移还在继续）
- 极低噪声（1e-5）时精度反而下降，原因不明
- 惯量失配不对称，低估 J 反而略好，原因不明

## 下一步

- 在真实的电机 + 驱动器上采数据，替换仿真数据源
- 力矩标定：砝码加杠杆臂做静态标定
- 温度补偿改成 d 轴电流注入，在线辨识 Rs 和磁链

## 参考

De Luca & Mattone 的广义动量观测器；Canudas de Wit 的 LuGre 摩擦模型。

## License

MIT
