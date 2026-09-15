"""仿真主程序：验证"自由段自监督辨识 + 接触段力矩估计"闭环。

运行:
    python src/simulate.py
输出:
    docs/figures/*.png  四张图
    docs/figures/metrics.txt  关键指标

仿真场景:
    0 ~ 2.0 s   自由运动（tau_ext = 0），多正弦轨迹激励，RLS 在线辨识摩擦参数
    2.0 ~ 4.0 s 接触段，施加已知外部力矩，用 tau_ext_hat = r - tau_f_hat 估计它

⚠️ 状态说明：本脚本是本项目的**可运行原型骨架**。
   请在本地运行并核对数值；若与预期不符，优先检查：
   (1) K_O*Ts 是否满足稳定性条件  (2) 测速噪声量级是否过大
   (3) RLS 的 P0 与遗忘因子 lam 是否导致发散
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joint_model import (  # noqa: E402
    JointParams,
    external_torque,
    friction_torque,
    reference_trajectory,
)
from friction_rls import FrictionRLS, grid_search_vs  # noqa: E402
from momentum_observer import MomentumObserver  # noqa: E402

# ---------------------------------------------------------------- 配置
Ts = 1e-4                 # 采样周期 [s]
T_END = 4.0               # 总时长 [s]
T_CONTACT = 2.0           # 接触段开始时刻 [s]
K_O = 80.0                # 观测器增益 [1/s]
KP, KD = 18.0, 0.9        # 位置环 PD 增益
I_LIMIT = 12.0            # 电流限幅 [A]
VEL_NOISE_STD = 5e-4      # 测速噪声标准差 [rad/s]
VEL_LPF_TAU = 2e-3        # 测速低通滤波时间常数 [s]
SEED = 0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "docs", "figures")


def main() -> None:
    rng = np.random.default_rng(SEED)
    p = JointParams()
    obs = MomentumObserver(J=p.J, K_O=K_O, Ts=Ts)
    rls = FrictionRLS(v_s=p.v_s * 0.6)   # 故意给一个偏差的初值，考察收敛能力

    n = int(T_END / Ts)
    q, qd, qd_meas = 0.0, 0.0, 0.0
    alpha = Ts / (VEL_LPF_TAU + Ts)

    log = {k: np.zeros(n) for k in
           ("t", "q", "q_ref", "qd", "tau_m", "r", "tau_f", "tau_ext",
            "tau_ext_hat", "tau_c_hat", "tau_s_hat", "b_hat")}

    for k in range(n):
        t = k * Ts
        q_ref, qd_ref = reference_trajectory(t)
        tau_ext = external_torque(t, T_CONTACT)

        # --- 控制（位置环 PD，输出电流指令）---
        i_q = KP * (q_ref - q) + KD * (qd_ref - qd)
        i_q = float(np.clip(i_q, -I_LIMIT, I_LIMIT))
        tau_m = p.Kt * i_q

        # --- 观测器（只用测量量：qd_meas 与 tau_m）---
        r = obs.step(qd_meas, tau_m)

        # --- 自由段在线辨识 ---
        if t < T_CONTACT:
            rls.update(qd_meas, r)

        tau_f = float(friction_torque(qd, p))
        tau_ext_hat = r - rls.predict(qd_meas)   # 接触段力矩估计

        # --- 记录 ---
        log["t"][k] = t
        log["q"][k] = q
        log["q_ref"][k] = q_ref
        log["qd"][k] = qd
        log["tau_m"][k] = tau_m
        log["r"][k] = r
        log["tau_f"][k] = tau_f
        log["tau_ext"][k] = tau_ext
        log["tau_ext_hat"][k] = tau_ext_hat
        log["tau_c_hat"][k] = rls.tau_c
        log["tau_s_hat"][k] = rls.tau_s
        log["b_hat"][k] = rls.b

        # --- 积分（半隐式欧拉）---
        acc = (tau_m - tau_f - p.tau_g - tau_ext) / p.J
        qd += acc * Ts
        q += qd * Ts

        # --- 测速：编码器差分 + 噪声 + 低通（真实系统的样子）---
        qd_meas += alpha * ((qd + rng.normal(0.0, VEL_NOISE_STD)) - qd_meas)

    # ------------------------------------------------------------ 指标
    free = log["t"] < T_CONTACT
    vs_fit, rmse_fit = grid_search_vs(log["qd"][free], log["r"][free])

    contact = ~free
    err = log["tau_ext_hat"][contact] - log["tau_ext"][contact]
    rmse_ext = float(np.sqrt(np.mean(err ** 2)))
    mae_ext = float(np.mean(np.abs(err)))
    denom = float(np.sqrt(np.mean(log["tau_ext"][contact] ** 2)))
    nrmse_ext = rmse_ext / denom if denom > 0 else float("nan")

    # 基线：只用开环电流估计力矩（即忽略摩擦）
    base_err = log["tau_m"][contact] - log["tau_ext"][contact]
    base_rmse = float(np.sqrt(np.mean(base_err ** 2)))

    lines = [
        "=== 摩擦辨识（自由段）===",
        f"真值      : tau_c={p.tau_c:.4f}  tau_s={p.tau_s:.4f}  b={p.b:.4f}  v_s={p.v_s:.4f}",
        f"RLS 辨识  : tau_c={rls.tau_c:.4f}  tau_s={rls.tau_s:.4f}  b={rls.b:.4f}"
        f"  (v_s 固定为 {rls.v_s:.4f})",
        f"网格搜索  : v_s={vs_fit:.4f}  RMSE={rmse_fit:.5f}",
        "",
        "=== 接触段力矩估计误差 ===",
        f"本方法 RMSE = {rmse_ext:.5f} N*m   MAE = {mae_ext:.5f}   nRMSE = {nrmse_ext:.2%}",
        f"基线(纯电流) RMSE = {base_rmse:.5f} N*m",
        f"相对基线改善 = {(1 - rmse_ext / base_rmse):.1%}" if base_rmse > 0 else "",
    ]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ------------------------------------------------------------ 绘图
    t = log["t"]

    plt.figure(figsize=(9, 3.2))
    plt.plot(t, log["q_ref"], "k--", lw=1, label="q_ref")
    plt.plot(t, log["q"], "b", lw=1.2, label="q")
    plt.axvline(T_CONTACT, color="r", ls=":", lw=1)
    plt.xlabel("t [s]"); plt.ylabel("q [rad]"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Position tracking"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig1_tracking.png"), dpi=150); plt.close()

    plt.figure(figsize=(9, 3.2))
    plt.plot(t, log["tau_f"], "k", lw=1.2, label="true friction")
    plt.plot(t, log["r"], "r", lw=0.8, alpha=.8, label="observer residual r")
    plt.axvline(T_CONTACT, color="r", ls=":", lw=1)
    plt.xlabel("t [s]"); plt.ylabel("torque [N*m]"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Free phase: r tracks the true friction"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig2_residual_vs_friction.png"),
                dpi=150); plt.close()

    plt.figure(figsize=(9, 3.2))
    plt.plot(t, log["tau_c_hat"], label="tau_c_hat")
    plt.plot(t, log["tau_s_hat"], label="tau_s_hat")
    plt.plot(t, log["b_hat"], label="b_hat")
    plt.axhline(p.tau_c, color="C0", ls=":", lw=1)
    plt.axhline(p.tau_s, color="C1", ls=":", lw=1)
    plt.axhline(p.b, color="C2", ls=":", lw=1)
    plt.axvline(T_CONTACT, color="r", ls=":", lw=1)
    plt.xlabel("t [s]"); plt.ylabel("param"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Online friction identification (dotted = true value)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig3_param_convergence.png"),
                dpi=150); plt.close()

    plt.figure(figsize=(9, 3.2))
    plt.plot(t, log["tau_ext"], "k", lw=1.5, label="true external torque")
    plt.plot(t, log["tau_ext_hat"], "r--", lw=1.2, label="estimated")
    plt.axvline(T_CONTACT, color="r", ls=":", lw=1)
    plt.xlabel("t [s]"); plt.ylabel("torque [N*m]"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Contact phase: sensorless external torque estimation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig4_external_torque.png"),
                dpi=150); plt.close()

    print(f"\n图已保存至 {os.path.normpath(OUT_DIR)}")


if __name__ == "__main__":
    main()
