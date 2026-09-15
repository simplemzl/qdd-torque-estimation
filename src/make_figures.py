"""零依赖 SVG 绘图：生成论文用图。

【为什么自己写绘图器】
  simulate.py 用 numpy + matplotlib，但本机装不上（无网络），
  用户的机器也未必有。所以这里用**纯标准库直接生成 SVG**：
    · 任何 Python 3 都能跑，不需要任何第三方库
    · SVG 是矢量图，可直接进论文、可无损缩放
    · 可在浏览器里打开检查
  输出同时验证为合法 XML（用 xml.etree 解析一遍）。

【生成的图】
  fig1_observer.svg    自由段残差 vs 真实摩擦；接触段力矩估计对比
  fig2_rmse_bars.svg   四种估计方式的接触段 RMSE 对比
  fig3_cross_task.svg  4 个任务 × 3 种辨识方式
  fig4_error_budget.svg 误差预算：齿槽/温漂，补偿前后

运行:
    python src/make_figures.py
"""
from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import error_budget as EB  # noqa: E402
import regime_study as RS  # noqa: E402
import simulate_stdlib as S  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "docs", "figures")

# 配色（对色盲友好）
C_TRUE = "#111111"
C_EST = "#d62728"
C_ALT = "#1f77b4"
C_ORACLE = "#7f7f7f"
C_OK = "#2ca02c"
C_WARN = "#ff7f0e"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_ticks(lo: float, hi: float, n: int = 5):
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    else:
        step = 10 * mag
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + step * 1e-9:
        out.append(round(t, 12))
        t += step
    return out


class Plot:
    """极简 SVG 折线/柱状图绘制器。"""

    def __init__(self, w=900, h=520, ml=78, mr=28, mt=46, mb=64):
        self.w, self.h = w, h
        self.ml, self.mr, self.mt, self.mb = ml, mr, mt, mb
        self.xlim = (0.0, 1.0)
        self.ylim = (0.0, 1.0)
        self.el = []
        self.title = ""
        self.xlabel = ""
        self.ylabel = ""
        self._legend = []

    # --- 坐标变换 ---
    def sx(self, x):
        a, b = self.xlim
        return self.ml + (x - a) / (b - a) * (self.w - self.ml - self.mr)

    def sy(self, y):
        a, b = self.ylim
        return self.h - self.mb - (y - a) / (b - a) * (self.h - self.mt - self.mb)

    # --- 图元 ---
    def line(self, x1, y1, x2, y2, color="#999", w=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.el.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
                       f'y2="{y2:.2f}" stroke="{color}" stroke-width="{w}"{d}/>')

    def poly(self, pts, color=C_EST, w=1.8, dash=None):
        if len(pts) < 2:
            return
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{self.sx(x):.2f},{self.sy(y):.2f}" for x, y in pts)
        self.el.append(f'<polyline points="{p}" fill="none" stroke="{color}" '
                       f'stroke-width="{w}" stroke-linejoin="round"{d}/>')

    def rect(self, x, y, w_, h_, fill, stroke="none", sw=1.0):
        self.el.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w_:.2f}" '
                       f'height="{h_:.2f}" fill="{fill}" stroke="{stroke}" '
                       f'stroke-width="{sw}"/>')

    def text(self, x, y, s, size=13, anchor="start", color="#222", weight="normal"):
        self.el.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
                       f'font-family="Helvetica,Arial,sans-serif" '
                       f'text-anchor="{anchor}" fill="{color}" '
                       f'font-weight="{weight}">{esc(s)}</text>')

    # --- 装饰 ---
    def draw_axes(self, xlabel=None, ylabel=None, title=None, xfmt="{:.3g}",
                  yfmt="{:.3g}", xrot=0):
        self.el.append(f'<rect x="{self.ml}" y="{self.mt}" '
                       f'width="{self.w - self.ml - self.mr}" '
                       f'height="{self.h - self.mt - self.mb}" fill="white" '
                       f'stroke="#333" stroke-width="1"/>')
        for t in nice_ticks(*self.xlim):
            X = self.sx(t)
            self.line(X, self.mt, X, self.h - self.mb, "#e6e6e6")
            self.line(X, self.h - self.mb, X, self.h - self.mb + 5, "#333")
            self.text(X, self.h - self.mb + 20, xfmt.format(t), 12, "middle")
        for t in nice_ticks(*self.ylim):
            Y = self.sy(t)
            self.line(self.ml, Y, self.w - self.mr, Y, "#e6e6e6")
            self.line(self.ml - 5, Y, self.ml, Y, "#333")
            self.text(self.ml - 9, Y + 4, yfmt.format(t), 12, "end")
        if xlabel:
            self.text((self.ml + self.w - self.mr) / 2, self.h - 16,
                      xlabel, 14, "middle")
        if ylabel:
            self.el.append(
                f'<text x="18" y="{(self.mt + self.h - self.mb) / 2:.1f}" '
                f'font-size="14" font-family="Helvetica,Arial,sans-serif" '
                f'text-anchor="middle" fill="#222" '
                f'transform="rotate(-90 18 {(self.mt + self.h - self.mb) / 2:.1f})">'
                f'{esc(ylabel)}</text>')
        if title:
            self.text((self.ml + self.w - self.mr) / 2, 26, title, 16, "middle",
                      "#111", "bold")

    def add_legend(self, items, x=None, y=None):
        """items: [(label, color, dash or None), ...]"""
        bw = 250
        bh = 22 * len(items) + 12
        X = x if x is not None else self.w - self.mr - bw - 14
        Y = y if y is not None else self.mt + 14
        self.rect(X, Y, bw, bh, "white", "#bbb", 1)
        for i, (lab, col, dash) in enumerate(items):
            yy = Y + 20 + i * 22
            d = f' stroke-dasharray="{dash}"' if dash else ""
            self.el.append(f'<line x1="{X + 12}" y1="{yy - 4}" x2="{X + 46}" '
                           f'y2="{yy - 4}" stroke="{col}" stroke-width="2.4"{d}/>')
            self.text(X + 54, yy, lab, 12)

    def save(self, path):
        body = "\n".join(self.el)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
               f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">\n'
               f'<rect width="{self.w}" height="{self.h}" fill="white"/>\n'
               f'{body}\n</svg>\n')
        # 合法性检查：必须是可解析的 XML
        ET.fromstring(svg)
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        return path


# =============================================================== 数据采集
def collect_traces():
    """跑一次仿真，密集记录自由段与接触段，用于 fig1。"""
    KO = 200.0
    rng_free, rng_contact = [], []
    model = None
    import random
    rng = random.Random(0)
    alpha = S.TS / (S.VEL_LPF_TAU + S.TS)
    WARM, CONTACT = RS.WARM, RS.CONTACT
    n = int((WARM + CONTACT) / S.TS)
    q = qd = qdm = 0.0
    prev, r = None, 0.0
    samples = []

    for k in range(n):
        t = k * S.TS
        if t < WARM:
            qr, qdr = RS.task_t1(t)
            te = 0.0
        else:
            qr, qdr = RS.task_t1(t - WARM)
            te = RS.TAU_EXT_AMP * math.sin(2 * math.pi * 1.0 * (t - WARM))
        iq = max(-S.I_LIMIT, min(S.I_LIMIT, S.KP * (qr - q) + S.KD * (qdr - qd)))
        tm = S.KT * iq
        if prev is None:
            prev = qdm
        else:
            r += KO * S.TS * (tm - S.J * (qdm - prev) / S.TS - r)
            prev = qdm
        if t < WARM:
            if abs(qdm) >= 1e-3 and k % 20 == 0:
                samples.append((qdm, r))
            if k % 200 == 0:
                rng_free.append((t, S.friction_true(qd), r))
        else:
            if model is None:
                model = S.batch_fit(samples)
            vs, th = model[0], model[1]
            tf = sum(S.phi_vec(qdm, vs)[i] * th[i] for i in range(3))
            if k % 200 == 0:
                rng_contact.append((t - WARM, te, r - tf, r,
                                    r - S.friction_true(qdm)))
        acc = (tm - S.friction_true(qd) - te) / S.J
        qd += acc * S.TS
        q += qd * S.TS
        qdm += alpha * ((qd + rng.gauss(0.0, S.VEL_NOISE_STD)) - qdm)
    return rng_free, rng_contact


# =============================================================== 图 1
def fig1():
    free, cont = collect_traces()

    sub = Plot(920, 330, ml=80, mr=30, mt=46, mb=58)
    sub.xlim = (0, max(t for t, _, _ in free))
    lo = min(min(a, b) for _, a, b in free)
    hi = max(max(a, b) for _, a, b in free)
    pad = (hi - lo) * 0.12 + 1e-6
    sub.ylim = (lo - pad, hi + pad)
    sub.draw_axes("time [s]", "torque [N*m]",
                  "Free motion: observer residual tracks true friction")
    sub.poly([(t, a) for t, a, _ in free], C_TRUE, 2.0)
    sub.poly([(t, b) for t, _, b in free], C_EST, 1.6)
    sub.add_legend([("true friction", C_TRUE, None),
                    ("observer residual r", C_EST, None)])
    sub.save(os.path.join(OUT, "_f1a.svg"))

    sub2 = Plot(920, 330, ml=80, mr=30, mt=46, mb=58)
    sub2.xlim = (0, max(t for t, *_ in cont))
    vals = [v for _, a, b, c, d in cont for v in (a, b, c, d)]
    pad = (max(vals) - min(vals)) * 0.12 + 1e-6
    sub2.ylim = (min(vals) - pad, max(vals) + pad)
    sub2.draw_axes("time [s]", "torque [N*m]",
                   "Contact: sensorless external-torque estimation")
    sub2.poly([(t, a) for t, a, *_ in cont], C_TRUE, 2.2)
    sub2.poly([(t, b) for t, _, b, *_ in cont], C_EST, 1.8)
    sub2.poly([(t, c) for t, _, _, c, _ in cont], C_ALT, 1.4, dash="5,4")
    sub2.add_legend([("true external torque", C_TRUE, None),
                     ("estimated (identified friction)", C_EST, None),
                     ("residual w/o friction comp.", C_ALT, "5,4")])
    sub2.save(os.path.join(OUT, "_f1b.svg"))
    return os.path.join(OUT, "_f1a.svg"), os.path.join(OUT, "_f1b.svg")


# =============================================================== 图 2
def fig2():
    r = S.run_sim("A")
    vals = [("pure current\nbaseline", r["rmse_base"], C_ORACLE),
            ("residual only\n(no friction comp.)", r["rmse_ref"], C_ALT),
            ("identified friction\n(this method)", r["rmse_ext"], C_OK),
            ("oracle\n(true friction)", r["rmse_perfect"], C_WARN)]
    p = Plot(820, 500, mb=86)
    p.xlim = (0, len(vals))
    p.ylim = (0, max(v for _, v, _ in vals) * 1.18)
    p.draw_axes(None, "RMSE [N*m]",
                "Contact-phase torque estimation error (task T1)", xfmt="{:.0f}",
                yfmt="{:.3f}")
    bw = 0.52
    for i, (lab, v, col) in enumerate(vals):
        x = i + 0.5 - bw / 2
        y0, y1 = p.sy(0), p.sy(v)
        p.rect(p.sx(x), y1, p.sx(x + bw) - p.sx(x), y0 - y1, col)
        p.text(p.sx(i + 0.5), y1 - 8, f"{v:.4f}", 13, "middle", "#111", "bold")
        for j, line in enumerate(lab.split("\n")):
            p.text(p.sx(i + 0.5), p.h - 56 + j * 16, line, 12, "middle")
    return p.save(os.path.join(OUT, "fig2_rmse_bars.svg"))


# 图 3 用的英文任务标签（论文是英文，且 SVG 字体无中文字形）
TASK_LABEL = {
    "T1 多正弦0.3/1.1": "T1  multisine 0.3/1.1 Hz",
    "T2 单频0.5": "T2  single sine 0.5 Hz",
    "T3 慢多正弦0.1/0.7": "T3  multisine 0.1/0.7 Hz",
    "T4 调频0.2-1.5": "T4  chirp 0.2-1.5 Hz",
}


# =============================================================== 图 3
def fig3():
    tasks = list(RS.TASKS.items())
    methods = [("in-regime", None, C_OK),
               ("low-speed protocol", RS.PROTO_B, C_WARN),
               ("full-range protocol", RS.PROTO_C, C_ALT)]
    data = {}
    for tname, tfn in tasks:
        for mname, proto, _ in methods:
            if proto is None:
                r = RS.run_case(RS.in_regime_traj(tfn), RS.WARM, tfn, 200.0)
            else:
                tot = RS.PROTO_B_T if proto is RS.PROTO_B else RS.PROTO_C_T
                r = RS.run_case(proto, tot, tfn, 200.0)
            data[(tname, mname)] = r["rmse"]

    p = Plot(1000, 520, mb=92, ml=86)
    p.xlim = (0, len(tasks))
    p.ylim = (0, max(data.values()) * 1.55)      # 留出图例空间，避免压柱
    p.draw_axes(None, "RMSE [N*m]",
                "Identification regime vs. accuracy, across 4 tasks",
                xfmt="{:.0f}", yfmt="{:.3f}")
    gw = 0.8 / len(methods)
    for i, (tname, _) in enumerate(tasks):
        for j, (mname, _, col) in enumerate(methods):
            v = data[(tname, mname)]
            x = i + 0.1 + j * gw
            p.rect(p.sx(x), p.sy(v), p.sx(x + gw * 0.9) - p.sx(x),
                   p.sy(0) - p.sy(v), col)
            p.text(p.sx(x + gw * 0.45), p.sy(v) - 7, f"{v:.3f}", 11, "middle")
        p.text(p.sx(i + 0.5), p.h - 56, TASK_LABEL.get(tname, tname),
               12.5, "middle")
    p.add_legend([(m, c, None) for m, _, c in methods],
                 x=p.w - 250, y=p.mt + 10)
    return p.save(os.path.join(OUT, "fig3_cross_task.svg"))


# =============================================================== 图 4
def fig4():
    rows = []
    for name, eff in EB.EFFECTS.items():
        a = EB.run_case(eff, comp_cog=False)
        b = EB.run_case(eff, comp_cog=True)
        rows.append((name.split()[0], a["rmse"], b["rmse"]))
    p = Plot(900, 520, mb=92, ml=86)
    p.xlim = (0, len(rows))
    p.ylim = (0, max(max(a, b) for _, a, b in rows) * 1.55)
    p.draw_axes(None, "RMSE [N*m]",
                "Error budget: cogging / temperature drift, before vs after compensation",
                xfmt="{:.0f}", yfmt="{:.3f}")
    gw = 0.34
    for i, (name, a, b) in enumerate(rows):
        x1 = i + 0.5 - gw - 0.02
        x2 = i + 0.5 + 0.02
        p.rect(p.sx(x1), p.sy(a), p.sx(x1 + gw) - p.sx(x1), p.sy(0) - p.sy(a), C_ORACLE)
        p.rect(p.sx(x2), p.sy(b), p.sx(x2 + gw) - p.sx(x2), p.sy(0) - p.sy(b), C_OK)
        p.text(p.sx(x1 + gw / 2), p.sy(a) - 7, f"{a:.4f}", 11, "middle")
        p.text(p.sx(x2 + gw / 2), p.sy(b) - 7, f"{b:.4f}", 11, "middle")
        p.text(p.sx(i + 0.5), p.h - 56, name, 12.5, "middle")
    p.add_legend([("no compensation", C_ORACLE, None),
                  ("with compensation", C_OK, None)], x=p.w - 250, y=p.mt + 10)
    return p.save(os.path.join(OUT, "fig4_error_budget.svg"))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    made = []
    print("生成图中...")
    made += list(fig1())
    made.append(fig2())
    print("  fig3 需要重跑跨任务实验，请稍候...")
    made.append(fig3())
    made.append(fig4())

    # fig1 的两半拼成一张纵向排布图
    combined = _combine_fig1(made[0], made[1])

    print("\n生成结果（全部通过 XML 合法性检查）:")
    for f in [combined] + made[2:]:
        print(f"  {os.path.basename(f)}  ({os.path.getsize(f)} bytes)")


def _combine_fig1(a_path, b_path):
    """把 fig1 的上下两半拼成一张竖排 SVG。"""
    A = ET.parse(a_path).getroot()
    B = ET.parse(b_path).getroot()
    w = 920
    ha = int(A.get("height"))
    hb = int(B.get("height"))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" '
           f'height="{ha + hb}" viewBox="0 0 {w} {ha + hb}">',
           f'<rect width="{w}" height="{ha + hb}" fill="white"/>']
    for child in list(A):
        tag = child.tag.split("}")[-1]
        if tag == "rect" and child.get("width") == str(w):
            continue
        out.append(ET.tostring(child, encoding="unicode"))
    out.append(f'<g transform="translate(0,{ha})">')
    for child in list(B):
        tag = child.tag.split("}")[-1]
        if tag == "rect" and child.get("width") == str(w):
            continue
        out.append(ET.tostring(child, encoding="unicode"))
    out.append("</g></svg>")
    svg = "\n".join(out)
    ET.fromstring(svg)
    path = os.path.join(OUT, "fig1_observer.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    for tmp in (a_path, b_path):
        try:
            os.remove(tmp)
        except OSError:
            pass
    return path


if __name__ == "__main__":
    main()
