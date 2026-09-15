"""摩擦参数在线辨识（递推最小二乘 RLS）。

对应论文第 3.2 节 —— 本文的核心创新点。

关键思路（"自监督"）：
    动量观测器残差 r 在自由运动段就是摩擦力 tau_f。
    也就是说，机器人在**正常运动**时，就免费获得了摩擦辨识的监督信号，
    不需要专门设计匀速/扫频辨识实验。

线性参数化（把 Stribeck 特征速度 v_s 当作固定超参数，其余参数线性）:

    tau_f = th1 * sgn(qd)
          + th2 * exp(-|qd|/v_s) * sgn(qd)
          + th3 * qd

    phi(qd) = [ sgn(qd), exp(-|qd|/v_s)*sgn(qd), qd ]
    theta   = [ tau_c, tau_s - tau_c, b ]

    => tau_c = th1,  tau_s = th1 + th2,  b = th3

RLS 更新:

    K_k     = P_{k-1} phi_k / (lam + phi_k^T P_{k-1} phi_k)
    theta_k = theta_{k-1} + K_k * (y_k - phi_k^T theta_{k-1})
    P_k     = (P_{k-1} - K_k phi_k^T P_{k-1}) / lam

工程上必须加的三个保护（否则参数会漂移/发散）：
    1. 速度死区：|qd| 太小时不更新（该处摩擦模型本身不可信）
    2. 残差野值抑制：|err| 过大时不更新（可能是碰撞或噪声）
    3. 协方差阵对称化：数值误差会破坏对称性，导致发散

v_s 的选取：先用网格搜索离线粗定，再固定进 RLS。
若要在线估计 v_s（模型对 v_s 非线性），需改用 EKF —— 见技术方案 4.3。
"""
from __future__ import annotations

import numpy as np


class FrictionRLS:
    def __init__(
        self,
        v_s: float = 0.05,
        lam: float = 0.999,
        theta0=None,
        P0: float = 1e2,
        vel_deadband: float = 1e-3,
        err_clip: float = 1.0,
    ):
        self.v_s = float(v_s)
        self.lam = float(lam)
        self.vel_deadband = float(vel_deadband)
        self.err_clip = float(err_clip)

        self.theta = np.zeros(3) if theta0 is None else np.asarray(theta0, float)
        self.P = np.eye(3) * float(P0)
        self.n_updates = 0

    def phi(self, qd: float) -> np.ndarray:
        s = np.sign(qd)
        return np.array([s, np.exp(-abs(qd) / self.v_s) * s, qd])

    def update(self, qd: float, y: float) -> np.ndarray:
        """用一次观测 (qd, y) 更新参数。y 应为动量观测器残差 r。"""
        if abs(qd) < self.vel_deadband:
            return self.theta

        ph = self.phi(qd)
        err = y - ph @ self.theta
        if abs(err) > self.err_clip:      # 野值抑制
            return self.theta

        Pph = self.P @ ph
        denom = self.lam + ph @ Pph
        if denom <= 1e-12:
            return self.theta

        K = Pph / denom
        self.theta = self.theta + K * err
        self.P = (self.P - np.outer(K, Pph)) / self.lam
        self.P = 0.5 * (self.P + self.P.T)   # 保对称
        self.n_updates += 1
        return self.theta

    def predict(self, qd: float) -> float:
        """用当前参数预测摩擦力。"""
        return float(self.phi(qd) @ self.theta)

    # --- 便于阅读的物理量访问器 ---
    @property
    def tau_c(self) -> float:
        return float(self.theta[0])

    @property
    def tau_s(self) -> float:
        return float(self.theta[0] + self.theta[1])

    @property
    def b(self) -> float:
        return float(self.theta[2])


def grid_search_vs(qd_data, y_data, v_s_grid=None) -> tuple[float, float]:
    """离线粗定 v_s：对每个候选 v_s 做最小二乘，取残差最小者。

    返回 (best_v_s, rmse)。RLS 之前先跑一次这个。
    """
    qd_data = np.asarray(qd_data, float)
    y_data = np.asarray(y_data, float)
    if v_s_grid is None:
        v_s_grid = np.logspace(-3, 0, 40)

    best_vs, best_rmse, best_theta = None, np.inf, None
    for vs in v_s_grid:
        s = np.sign(qd_data)
        Phi = np.column_stack([s, np.exp(-np.abs(qd_data) / vs) * s, qd_data])
        theta, *_ = np.linalg.lstsq(Phi, y_data, rcond=None)
        rmse = float(np.sqrt(np.mean((Phi @ theta - y_data) ** 2)))
        if rmse < best_rmse:
            best_vs, best_rmse, best_theta = float(vs), rmse, theta
    return best_vs, best_rmse
