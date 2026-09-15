"""广义动量观测器（Generalized Momentum Observer）。

对应论文第 3.1 节。核心思想：不用力矩传感器，只靠电流与编码器，
把"摩擦力 + 外部接触力矩"从动量残差里反解出来。

连续形式（De Luca & Mattone 结构，1 自由度简化）:

    p(t) = J * qd(t)                                    # 广义动量
    r(t) = K_O * [ p(t) - ∫ ( tau_m - r ) ds ]          # 等价微分形式见下

微分形式:

    r_dot = K_O * ( tau_m - p_dot - r )

代入真实动力学 p_dot = J*qdd = tau_m - tau_f - tau_ext，得

    r_dot = K_O * ( tau_f + tau_ext - r )

    =>  r 收敛到 (tau_f + tau_ext)

这个收敛性质是整个方法的基础：
    · 自由运动段（tau_ext = 0）:  r -> tau_f          => 可用于摩擦辨识
    · 接触段（tau_ext != 0）    :  r -> tau_f + tau_ext => tau_ext = r - tau_f_hat

离散实现:

    r_{k+1} = r_k + K_O * Ts * ( tau_m,k - (p_{k+1}-p_k)/Ts - r_k )

稳定性条件: 0 < K_O * Ts < 2。收敛时间常数约 1/K_O。
K_O 越大收敛越快，但对测速噪声越敏感 —— 这个权衡本身就是论文的实验点。
"""
from __future__ import annotations


class MomentumObserver:
    def __init__(self, J: float, K_O: float = 100.0, Ts: float = 1e-4):
        if not 0.0 < K_O * Ts < 2.0:
            raise ValueError(
                f"K_O*Ts = {K_O * Ts:.4f} 不满足稳定性条件 0 < K_O*Ts < 2"
            )
        self.J = float(J)
        self.K_O = float(K_O)
        self.Ts = float(Ts)
        self.reset()

    def reset(self) -> None:
        self._qd_prev = None
        self._r = 0.0

    def step(self, qd: float, tau_m: float) -> float:
        """推进一个采样步。

        Args:
            qd: 当前角速度测量值 [rad/s]（真实系统中来自编码器差分，含噪声）
            tau_m: 当前电机输出力矩 [N*m]，= Kt * i_q

        Returns:
            残差 r，收敛后等于 tau_f + tau_ext
        """
        if self._qd_prev is None:
            self._qd_prev = qd
            return self._r

        p_now = self.J * qd
        p_prev = self.J * self._qd_prev
        p_dot = (p_now - p_prev) / self.Ts

        self._r += self.K_O * self.Ts * (tau_m - p_dot - self._r)
        self._qd_prev = qd
        return self._r

    @property
    def residual(self) -> float:
        return self._r
