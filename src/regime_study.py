"""核心论点的稳健性检验：辨识工况与使用工况是否必须一致？

================================================================================
【待检验的论点】
上一轮在单一任务（多正弦）上发现：
    用「任务运动本身」做辨识（A 组）比用「标准低速辨识协议」（B 组）效果更好，
    力矩估计 RMSE 0.0302 vs 0.0505。
并据此推断机理：观测器残差是**经过观测器滤波的**摩擦力，辨识与补偿走同一
滤波器时系统性误差互相抵消；工况不匹配则这种抵消失效。

【问题】只有一个任务，结论不够硬。本脚本在 **4 个不同任务**上重复检验。

【实验设计】
对每个任务 T，用 4 种方式获得摩擦模型，然后**在同一个接触段任务上**评价：

  (i)  oracle     用真实摩擦模型（上界参考）
  (ii) in-regime  用任务 T 自身的前 4 秒做辨识     ← 本文主张
  (iii) proto-B   用标准低速恒速阶梯辨识（≤0.3 rad/s）
  (iv) proto-C    用全速域恒速阶梯辨识（≤2.0 rad/s）

评价指标：接触段外力矩估计 RMSE（t >= warm + 1s，剔除切换暂态）

【内置公平性检查】
所有方式的接触段参考轨迹完全相同，因此「纯电流基线」RMSE 应当完全一致。
若某个任务下基线不一致，说明该任务下仿真未收敛，结果不可用。

运行:
    python src/regime_study.py
"""
from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simulate_stdlib as S  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "docs", "figures")

WARM = 4.0          # 自由段时长 [s]
CONTACT = 3.0       # 接触段时长 [s]
SETTLE = 1.5        # 接触段暂态剔除 [s]（慢任务需要更长的同步时间）
TAU_EXT_AMP = 0.35


# ------------------------------------------------------------ 4 个任务
def task_t1(tau):
    """多正弦 0.3 + 1.1 Hz"""
    w1, w2 = 2 * math.pi * 0.30, 2 * math.pi * 1.10
    a1, a2 = 0.50, 0.20
    return (a1 * math.sin(w1 * tau) + a2 * math.sin(w2 * tau),
            a1 * w1 * math.cos(w1 * tau) + a2 * w2 * math.cos(w2 * tau))


def task_t2(tau):
    """单频 0.5 Hz（激励最贫乏）"""
    w, a = 2 * math.pi * 0.50, 0.60
    return a * math.sin(w * tau), a * w * math.cos(w * tau)


def task_t3(tau):
    """慢速多正弦 0.1 + 0.7 Hz"""
    w1, w2 = 2 * math.pi * 0.10, 2 * math.pi * 0.70
    a1, a2 = 0.70, 0.25
    return (a1 * math.sin(w1 * tau) + a2 * math.sin(w2 * tau),
            a1 * w1 * math.cos(w1 * tau) + a2 * w2 * math.cos(w2 * tau))


def task_t4(tau):
    """线性调频 0.2 -> 1.5 Hz（2 秒内）"""
    f0, f1, T = 0.20, 1.50, 2.0
    k = (f1 - f0) / T
    a = 0.50
    ph = 2 * math.pi * (f0 * tau + 0.5 * k * tau * tau)
    dph = 2 * math.pi * (f0 + k * tau)
    return a * math.sin(ph), a * dph * math.cos(ph)


TASKS = {"T1 多正弦0.3/1.1": task_t1,
         "T2 单频0.5": task_t2,
         "T3 慢多正弦0.1/0.7": task_t3,
         "T4 调频0.2-1.5": task_t4}


# --------------------------------------------------- 恒速阶梯协议
def make_staircase(segs):
    """返回 (traj_fn, total)。稳态窗取段内 60%~98%。"""
    knots, steady = [], []
    t = q = 0.0
    for v, dur in segs:
        knots.append((t, t + dur, q, v))
        steady.append((t + 0.60 * dur, t + 0.98 * dur))
        q += v * dur
        t += dur
    total = t

    def traj(t_):
        for t0, t1, q0, v in knots:
            if t0 <= t_ < t1:
                return q0 + v * (t_ - t0), v, any(a <= t_ <= b for a, b in steady)
        return knots[-1][2], 0.0, False

    return traj, total


PROTO_B, PROTO_B_T = make_staircase(S.SEGS_LOW)
PROTO_C, PROTO_C_T = make_staircase(S.SEGS_FULL)


def in_regime_traj(task_fn):
    """辨识用轨迹 = 任务轨迹本身（不做任何专门激励）"""
    def traj(t_):
        q, qd = task_fn(t_)
        return q, qd, True
    return traj


# ------------------------------------------------------------ 仿真核心
def run_case(free_fn, warm_end, task_fn, K_O, seed=0):
    """自由段用 free_fn（0..warm_end），接触段用 task_fn(t - warm_end)。"""
    t_end = warm_end + CONTACT
    rng = random.Random(seed)
    alpha = S.TS / (S.VEL_LPF_TAU + S.TS)
    n = int(t_end / S.TS)

    q = qd = qd_meas = 0.0
    qd_prev, r = None, 0.0
    samples = []
    se_ext = se_base = se_oracle = 0.0
    n_c = 0
    fit = None
    qd_min, qd_max = 0.0, 0.0

    for k in range(n):
        t = k * S.TS
        if t < warm_end:
            q_ref, qd_ref, steady = free_fn(t)
            tau_ext = 0.0
        else:
            q_ref, qd_ref = task_fn(t - warm_end)
            steady = False
            tau_ext = TAU_EXT_AMP * math.sin(2 * math.pi * 1.0 * (t - warm_end))

        i_q = max(-S.I_LIMIT, min(S.I_LIMIT,
                                  S.KP * (q_ref - q) + S.KD * (qd_ref - qd)))
        tau_m = S.KT * i_q

        if qd_prev is None:
            qd_prev = qd_meas
        else:
            r += K_O * S.TS * (tau_m - S.J * (qd_meas - qd_prev) / S.TS - r)
            qd_prev = qd_meas

        if t < warm_end:
            if steady and abs(qd_meas) >= 1e-3 and k % S.SAMPLE_STRIDE == 0:
                samples.append((qd_meas, r))
                qd_min = min(qd_min, qd_meas)
                qd_max = max(qd_max, qd_meas)
        else:
            if fit is None:
                fit = S.batch_fit(samples)
            vs_hat, th_hat = fit[0], fit[1]
            tau_hat = sum(S.phi_vec(qd_meas, vs_hat)[i] * th_hat[i]
                          for i in range(3))
            if t >= warm_end + SETTLE:
                se_ext += ((r - tau_hat) - tau_ext) ** 2
                se_base += (tau_m - tau_ext) ** 2
                se_oracle += ((r - S.friction_true(qd_meas)) - tau_ext) ** 2
                n_c += 1

        acc = (tau_m - S.friction_true(qd) - tau_ext) / S.J
        qd += acc * S.TS
        q += qd * S.TS
        qd_meas += alpha * ((qd + rng.gauss(0.0, S.VEL_NOISE_STD)) - qd_meas)

    vs_hat, th_hat = fit[0], fit[1]
    return {
        "rmse": math.sqrt(se_ext / n_c),
        "base": math.sqrt(se_base / n_c),
        "oracle": math.sqrt(se_oracle / n_c),
        "exc": (qd_min, qd_max),
        "n": len(samples),
        "xf": [th_hat[0], th_hat[0] + th_hat[1], th_hat[2], vs_hat],
    }


def curve_rmse(x, lo, hi, n=300):
    sse = 0.0
    for k in range(n):
        qd = lo + (hi - lo) * k / (n - 1)
        pred = sum(S.phi_vec(qd, x[3])[i] * [x[0], x[1] - x[0], x[2]][i]
                   for i in range(3))
        sse += (pred - S.friction_true(qd)) ** 2
    return math.sqrt(sse / n)


def main() -> None:
    KO = 200.0
    S.K_O = KO

    lines = [
        "=" * 82,
        "  核心论点稳健性检验：辨识工况 vs 使用工况（4 个任务 × 4 种辨识方式）",
        "=" * 82,
        "",
        f"  观测器 K_O = {KO:.0f} (时间常数 {1000 / KO:.2f} ms)",
        f"  真值: tau_c={S.TAU_C}  tau_s={S.TAU_S}  b={S.B_VISC}  v_s={S.V_S}",
        "",
    ]

    summary = []
    for tname, tfn in TASKS.items():
        cases = {
            "oracle": None,
            "in-regime": run_case(in_regime_traj(tfn), WARM, tfn, KO),
            "proto-B": run_case(PROTO_B, PROTO_B_T, tfn, KO),
            "proto-C": run_case(PROTO_C, PROTO_C_T, tfn, KO),
        }
        base_vals = [c["base"] for c in cases.values() if c]
        spread = (max(base_vals) - min(base_vals)) / (sum(base_vals) / len(base_vals))
        fair = spread < 1e-3        # 相对容差，避免浮点尾差误判

        bases = "  ".join(f"{k}={v['base']:.5f}" for k, v in cases.items() if v)
        lines.append(f"  【{tname}】  公平性检查: "
                     f"{'通过' if fair else '★不通过，结果不可用★'}")
        lines.append(f"    各方式基线(应完全一致): {bases}")
        lines.append(f"    {'辨识方式':<12}{'激励范围':>18}{'样本':>7}"
                     f"{'曲线RMSE':>10}{'力矩RMSE':>10}{'改善':>9}")
        lines.append(f"    {'oracle(真值)':<12}{'—':>18}{'—':>7}{'0':>10}"
                     f"{cases['in-regime']['oracle']:>10.5f}"
                     f"{1 - cases['in-regime']['oracle'] / cases['in-regime']['base']:>8.1%}")
        for key in ("in-regime", "proto-B", "proto-C"):
            c = cases[key]
            lo, hi = c["exc"]
            lines.append(
                f"    {key:<12}{f'[{lo:.2f},{hi:.2f}]':>18}{c['n']:>7}"
                f"{curve_rmse(c['xf'], -2.5, 2.5):>10.5f}{c['rmse']:>10.5f}"
                f"{1 - c['rmse'] / c['base']:>8.1%}")
        lines.append("")
        summary.append((tname, cases, fair))

    # ---- 汇总：in-regime 是否在 4 个任务上一致胜出 ----
    lines += ["=" * 82, "  汇总", "=" * 82, ""]
    lines.append(f"    {'任务':<20}{'in-regime':>12}{'proto-B':>11}"
                 f"{'proto-C':>11}{'oracle':>10}   胜出者")
    wins = 0
    for tname, c, fair in summary:
        vals = {"in-regime": c["in-regime"]["rmse"],
                "proto-B": c["proto-B"]["rmse"],
                "proto-C": c["proto-C"]["rmse"]}
        best = min(vals, key=vals.get)
        if best == "in-regime":
            wins += 1
        lines.append(f"    {tname:<20}{vals['in-regime']:>12.5f}"
                     f"{vals['proto-B']:>11.5f}{vals['proto-C']:>11.5f}"
                     f"{c['in-regime']['oracle']:>10.5f}   {best}")

    lines += [
        "",
        f"    in-regime 在 {wins}/{len(summary)} 个任务上取得最低力矩 RMSE",
        "",
        "  判读要点：",
        "    · oracle（用真实摩擦）通常不是最优 —— 这就是「完美摩擦悖论」",
        "      （详见 代码原型/FINDINGS.md）",
        "    · 若 in-regime 在多数任务上胜出，则「辨识工况需与使用工况一致」",
        "      这一论点具备跨任务稳健性，可作为论文核心贡献",
        "    · 若胜负混杂，则该论点只在特定任务成立，论文需收窄表述",
        "=" * 82,
    ]

    os.makedirs(OUT_DIR, exist_ok=True)
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT_DIR, "metrics_regime.txt"), "w",
              encoding="utf-8-sig") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
