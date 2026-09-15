"""摩擦参数辨识方法对比：网格最小二乘 vs 扩展卡尔曼滤波（EKF）。

【要回答的问题】
上一轮发现：Stribeck 参数辨识存在 (tau_s, v_s) 退化 —— 曲线形状拟合对了，
但参数值不对（v_s 总是跑到搜索边界）。
推测根因：把 v_s 当作**固定超参数**、在网格上离散选择，本身就丢掉了 v_s 的
连续可辨识性。

【本脚本做两件事】
1. 用 EKF 把 v_s 作为**在线估计的状态**，看参数退化是否改善。
2. 换一个更本质的评价指标：**力矩曲线精度**，而不只是参数精度。
   因为下游（摩擦前馈补偿）真正需要的是 tau_f(qd) 这条曲线准确，
   而不是 tau_c / tau_s / v_s 三个数各自准确。

评价指标：
    · 参数相对误差           —— 传统指标
    · 曲线 RMSE（全速域）     —— 下游真正关心的
    · 曲线 RMSE（低速/Stribeck区）—— 最难拟合的区间

运行:
    python src/identify_compare.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simulate_stdlib as S  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "docs", "figures")

# ------------------------------------------------------------------ EKF
# 状态 x = [tau_c, tau_s, b, v_s]
# 测量 y = r（观测器残差），在自由运动段 r -> tau_f
# 测量模型 h(x, qd) = [tau_c + (tau_s - tau_c)*exp(-|qd|/v_s)]*sgn(qd) + b*qd

X_MIN = [0.0, 0.0, -0.5, 1e-3]
X_MAX = [1.0, 1.0, 0.5, 1.0]


def h_and_H(x, qd):
    """返回 (h, H)，H 为 1x4 雅可比。"""
    tc, ts, b, vs = x
    s = S.sgn(qd)
    e = math.exp(-abs(qd) / vs)
    h = (tc + (ts - tc) * e) * s + b * qd
    d_tc = s * (1.0 - e)
    d_ts = s * e
    d_b = qd
    d_vs = s * (ts - tc) * e * abs(qd) / (vs * vs)
    return h, [d_tc, d_ts, d_b, d_vs]


def ekf_fit(samples, x0=None, P0=None, q_diag=None, R=1e-4):
    """对给定样本序列做 EKF 参数辨识（随机游走模型）。"""
    x = list(x0) if x0 else [0.10, 0.15, 0.005, 0.10]
    P = [row[:] for row in (P0 or [[1e-2, 0, 0, 0],
                                   [0, 1e-2, 0, 0],
                                   [0, 0, 1e-4, 0],
                                   [0, 0, 0, 1e-2]])]
    qd_q = q_diag or [1e-9, 1e-9, 1e-11, 1e-9]

    for qd, y in samples:
        # --- 预测（随机游走）---
        for i in range(4):
            P[i][i] += qd_q[i]

        # --- 更新（标量测量，无需矩阵求逆）---
        h, H = h_and_H(x, qd)
        PH = [sum(P[i][j] * H[j] for j in range(4)) for i in range(4)]
        denom = sum(H[i] * PH[i] for i in range(4)) + R
        if denom <= 1e-15:
            continue
        K = [PH[i] / denom for i in range(4)]
        innov = y - h
        # 野值抑制：创新过大时降权（可能是碰撞或暂态）
        if abs(innov) > 0.5:
            continue
        for i in range(4):
            x[i] += K[i] * innov
        # P = (I - K H) P
        newP = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                newP[i][j] = P[i][j] - K[i] * sum(H[m] * P[m][j] for m in range(4))
        P = [[0.5 * (newP[i][j] + newP[j][i]) for j in range(4)] for i in range(4)]

        # --- 物理约束 ---
        for i in range(4):
            x[i] = max(X_MIN[i], min(X_MAX[i], x[i]))
        # tau_s >= tau_c 是物理要求
        if x[1] < x[0]:
            x[1] = x[0]

    return x, P


# ------------------------------------------------------------ 评价指标
def curve_rmse(x, lo, hi, n=400):
    """在速度区间 [lo, hi] 上比较 tau_f 预测曲线与真值的 RMSE。"""
    pred = lambda qd: h_and_H(x, qd)[0]      # noqa: E731
    sse = 0.0
    for k in range(n):
        qd = lo + (hi - lo) * k / (n - 1)
        sse += (pred(qd) - S.friction_true(qd)) ** 2
    return math.sqrt(sse / n)


def param_errors(x):
    tc, ts, b, vs = x
    return (abs(tc - S.TAU_C) / S.TAU_C,
            abs(ts - S.TAU_S) / S.TAU_S,
            abs(b - S.B_VISC) / S.B_VISC,
            abs(vs - S.V_S) / S.V_S)


def main() -> None:
    lines = [
        "=" * 76,
        "  摩擦参数辨识：网格最小二乘 vs EKF —— 参数精度 vs 曲线精度",
        "=" * 76,
        "",
        f"  真值 : tau_c={S.TAU_C}  tau_s={S.TAU_S}  b={S.B_VISC}  v_s={S.V_S}",
        "",
    ]

    for grp in ("A", "C"):
        res = S.run_sim(grp)
        samples = res["samples"]

        # --- 方法 1：网格最小二乘 ---
        vs_ls, th_ls, rmse_ls, gd, rej = S.batch_fit(samples)
        x_ls = [th_ls[0], th_ls[0] + th_ls[1], th_ls[2], vs_ls]

        # --- 方法 2：EKF ---
        x_ekf, _ = ekf_fit(samples)

        lines.append(f"  【{grp} 组】 激励范围 [{res['exc_lo']:.2f}, {res['exc_hi']:.2f}]  "
                     f"rad/s，样本 {len(samples)}")
        lines.append(f"    {'方法':<14}{'tau_c':>9}{'tau_s':>9}{'b':>9}{'v_s':>9}"
                     f"{'曲线RMSE':>10}{'曲线RMSE':>11}{'低速RMSE':>10}")
        lines.append(f"    {'':<14}{'':>9}{'':>9}{'':>9}{'':>9}"
                     f"{'(全速域)':>10}{'(0.6~2.5)':>11}{'(<0.3)':>10}")

        for label, x in (("网格最小二乘", x_ls), ("EKF", x_ekf)):
            lines.append(
                f"    {label:<14}{x[0]:>9.4f}{x[1]:>9.4f}{x[2]:>9.4f}{x[3]:>9.4f}"
                f"{curve_rmse(x, -2.5, 2.5):>10.5f}"
                f"{curve_rmse(x, 0.6, 2.5):>11.5f}"
                f"{curve_rmse(x, -0.3, 0.3):>10.5f}")

        lines.append("")
        for label, x in (("网格最小二乘", x_ls), ("EKF", x_ekf)):
            e = param_errors(x)
            lines.append(f"    {label:<14} 参数相对误差: "
                         f"tau_c {e[0]:>6.1%}  tau_s {e[1]:>6.1%}  "
                         f"b {e[2]:>6.1%}  v_s {e[3]:>6.1%}")
        lines.append("")

    lines += [
        "=" * 76,
        "  怎么读这张表",
        "=" * 76,
        "  · 「曲线RMSE」是下游真正关心的指标 —— 摩擦前馈补偿用的是曲线，",
        "    不是 tau_c/tau_s/v_s 这三个数本身。",
        "  · 如果出现了「参数误差大、但曲线 RMSE 小」，说明参数退化对下游无害，",
        "    这本身就是一个有价值的结论（可以据此放宽对参数物理正确性的要求）。",
        "  · 「低速RMSE」是最难拟合的区间（Stribeck 转折处），也是误差集中区。",
        "=" * 76,
    ]

    os.makedirs(OUT_DIR, exist_ok=True)
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT_DIR, "metrics_identify.txt"), "w",
              encoding="utf-8-sig") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
