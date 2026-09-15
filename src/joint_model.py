"""单关节机电耦合模型（dq 轴简化 + Stribeck 摩擦）。

对应论文第 2 章。这是"真实系统"，用来做仿真验证；
辨识算法（momentum_observer.py / friction_rls.py）不允许看到这里的真参数。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class JointParams:
    """关节参数（仿真用真值）。"""

    J: float = 0.01        # 等效转动惯量 [kg*m^2]
    Kt: float = 0.50       # 转矩常数 [N*m/A]
    tau_c: float = 0.15    # 库仑摩擦 [N*m]
    tau_s: float = 0.25    # 静摩擦（Stribeck 低速峰值）[N*m]
    v_s: float = 0.05      # Stribeck 特征速度 [rad/s]
    b: float = 0.02        # 黏滞摩擦系数 [N*m*s/rad]
    tau_g: float = 0.0     # 常值负载 / 重力项 [N*m]


def friction_torque(qd, p: JointParams):
    """Stribeck + 黏滞摩擦模型。

        tau_f(qd) = [tau_c + (tau_s - tau_c) * exp(-|qd|/v_s)] * sgn(qd) + b*qd

    注意：该模型在 qd = 0 处 sgn(0) = 0，**不刻画静摩擦/预滑移**。
    这是刻意的简化，论文里要在"局限性"一节说明；
    更精确的 LuGre 模型作为对照实验（见技术方案 4.2）。
    """
    qd = np.asarray(qd, dtype=float)
    sgn = np.sign(qd)
    stribeck = p.tau_c + (p.tau_s - p.tau_c) * np.exp(-np.abs(qd) / p.v_s)
    return stribeck * sgn + p.b * qd


def acceleration(qd, i_q, tau_ext, p: JointParams):
    """由单关节动力学求角加速度。

        J * qdd = Kt * i_q - tau_f(qd) - tau_g - tau_ext
    """
    return (p.Kt * i_q - friction_torque(qd, p) - p.tau_g - tau_ext) / p.J


def reference_trajectory(t: float) -> tuple[float, float]:
    """多正弦参考轨迹：保证速度激励充分，使摩擦参数可辨识。

    单一频率的正弦激励会导致回归向量病态（某些参数不可辨识），
    多正弦叠加可以改善激励条件 —— 这一点本身值得在论文里讨论。
    """
    w1, w2 = 2 * np.pi * 0.30, 2 * np.pi * 1.10
    a1, a2 = 0.50, 0.20
    q_ref = a1 * np.sin(w1 * t) + a2 * np.sin(w2 * t)
    qd_ref = a1 * w1 * np.cos(w1 * t) + a2 * w2 * np.cos(w2 * t)
    return q_ref, qd_ref


def external_torque(t: float, t_contact_start: float) -> float:
    """外部接触力矩：前段为 0（自由运动，用于辨识），后段施加已知力矩。"""
    if t < t_contact_start:
        return 0.0
    return 0.35 * np.sin(2 * np.pi * 1.0 * (t - t_contact_start))
