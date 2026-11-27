import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

import analysis as A

BASE_DIR = Path(__file__).resolve().parent
LEDGER_DIR = BASE_DIR / "results" / "ledger"
FIG_DIR = BASE_DIR / "results" / "figures"
SUMMARY_PATH = BASE_DIR / "results" / "summary.txt"


def load(exp):
    return A.read_ledger(LEDGER_DIR / f"{exp}.jsonl")


def proper(r):
    return A.joints_proper(r["joint_AB"], r["joint_BA"])


def q_qq(r):
    return (r["joint_AB"][0] + r["joint_AB"][3]) - (r["joint_BA"][0] + r["joint_BA"][3])


def xy_pair(r):
    return (r["joint_AB"][0] - r["joint_BA"][0], r["joint_AB"][3] - r["joint_BA"][3])


def summarize(rows):
    q = np.array([q_qq(r) for r in rows])
    oss = np.array([abs(r["q_A"]) + abs(r["q_B"]) for r in rows])
    rec = np.array([r["q_A"] - r["q_B"] for r in rows])
    qA = np.array([r["q_A"] for r in rows])
    qB = np.array([r["q_B"] for r in rows])
    xya = np.array([xy_pair(r) for r in rows])
    corr = float(np.corrcoef(qA, qB)[0, 1]) if qA.std() > 1e-12 and qB.std() > 1e-12 else float("nan")
    slope = intercept = r_xy = float("nan")
    if xya[:, 0].std() > 1e-12:
        slope, intercept = np.polyfit(xya[:, 0], xya[:, 1], 1)
        r_xy = float(np.corrcoef(xya[:, 0], xya[:, 1])[0, 1])
    cbd_ok = [A.cbd2_lp(r["joint_AB"], r["joint_BA"])["feasible"] for r in rows]
    ceil = 4.0 / (3.0 * math.sqrt(3.0))
    ellipse_infeas = 0
    for r in rows:
        qa_i, qb_i = r["q_A"], r["q_B"]
        res = minimize_scalar(
            lambda g: qa_i * qa_i + qb_i * qb_i + 2.0 * g * qa_i * qb_i - ((1.0 - g * g) ** 2) / 4.0,
            bounds=(-1.0, 1.0),
            method="bounded",
        )
        if res.fun > 0.0:
            ellipse_infeas += 1
    return dict(
        n=len(rows),
        qQQ_median=float(np.median(q)),
        qQQ_absmedian=float(np.median(np.abs(q))),
        frac_band=float(np.mean(np.abs(q) <= 0.02)),
        frac_violation=float(np.mean(np.abs(q) > 0.05)),
        OSS_mean=float(oss.mean()),
        OSS_max=float(oss.max()),
        frac_above_ceil=float(np.mean(oss > ceil)),
        frac_ellipse_infeas=ellipse_infeas / len(rows),
        band=float(np.mean((np.abs(q) <= 0.02) & (oss >= 0.05))),
        R_absmedian=float(np.median(np.abs(rec))),
        corr_qA_qB=corr,
        xy_slope=float(slope),
        xy_intercept=float(intercept),
        xy_r=r_xy,
        xy_scale=float(np.median(np.abs(xya[:, 0]))),
        Gamma_frac_contextual=float(np.mean(np.abs(q) - oss > 1e-9)),
        cbd2_noncontextual_rate=float(np.mean(cbd_ok)),
    )


def e1_table():
    rows_all = load("E1")
    out = {}
    for s in ["luders_qubit", "lin_classical", "nl_classical", "qreservoir"]:
        rows = [r for r in rows_all if r["substrate"] == s]
        if s == "lin_classical":
            rows = [r for r in rows if proper(r)]
        out[s] = summarize(rows)
    return out


def e5_table():
    rows = load("E5")
    return {c: summarize([r for r in rows if r["cell"] == c]) for c in sorted({r["cell"] for r in rows})}


def e4_curves():
    rows = load("E4")
    curves = {"qres_dyn": {}, "qres_nodyn": {}, "nlc": {}}
    for r in rows:
        cell = r["cell"]
        if r["substrate"] == "qreservoir":
            if cell.startswith("p"):
                curves["qres_dyn"].setdefault(float(cell[1:]), []).append(abs(q_qq(r)))
            elif cell.startswith("nodyn_p"):
                curves["qres_nodyn"].setdefault(float(cell[7:]), []).append(abs(q_qq(r)))
        elif r["substrate"] == "nl_classical":
            curves["nlc"].setdefault(float(cell[1:]), []).append(abs(q_qq(r)))
    return {
        k: {p: (float(np.median(v)), float(np.percentile(v, 75)), float(np.mean(np.abs(v)))) for p, v in sorted(d.items())}
        for k, d in curves.items()
    }


def e2_table():
    rows = load("E2")
    out = {}
    for sub, cell in sorted({(r["substrate"], r["cell"]) for r in rows}):
        rs = [r for r in rows if r["substrate"] == sub and r["cell"] == cell]
        out[f"{sub}/{cell}"] = dict(
            n=len(rs),
            lp_feasible_rate=float(np.mean([r["lp_feasible"] for r in rs])),
            eps_star_median=float(np.median([r["eps_star"] for r in rs])),
            eps_star_q90=float(np.percentile([r["eps_star"] for r in rs], 90)),
            cbd_feasible_rate=float(np.mean([r["cbd_feasible"] for r in rs])),
            cbd_eps_median=float(np.median([r["cbd_eps"] for r in rs])),
            disturbance_mean=float(np.mean([r["disturbance_mag"] for r in rs])),
        )
    return out


def e6_table():
    rows = load("E6")
    out = {}
    for r in rows:
        out.setdefault(r["cell"], []).append(r)
    summ = {}
    for cell, rs in sorted(out.items()):
        q = np.array([abs(r["q_QQ"]) for r in rs])
        summ[cell] = dict(n=len(rs), qQQ_absmed=float(np.median(q)), qQQ_iqr75=float(np.percentile(q, 75)))
    by_cell = {c: rs for c, rs in out.items() if c.startswith("dyn0_n1_")}
    xs, ys = [], []
    for cell, rs in sorted(by_cell.items()):
        pu = float(cell.split("_pu")[1].split("_")[0])
        pd = float(cell.split("_pd")[1])
        xs.append(abs(pu - pd))
        ys.append(float(np.mean([abs(r["q_QQ"]) for r in rs])))
    r_lin = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 1 and np.std(xs) > 0 else float("nan")
    return summ, r_lin


def e7_table():
    rows = load("E7")
    out = {}
    for r in rows:
        out.setdefault(r["cell"], []).append(r)
    summ = {}
    for cell, rs in sorted(out.items()):
        R = np.array([min(r["R_rep_A"], r["R_rep_B"]) for r in rs])
        oss = np.array([r["OSS"] for r in rs])
        q = np.array([abs(r["q_QQ"]) for r in rs])
        summ[cell] = dict(
            n=len(rs),
            R_med=float(np.median(R)),
            OSS_mean=float(oss.mean()),
            qQQ_absmed=float(np.median(q)),
            triad=bool(np.mean(oss >= 0.05) > 0.5 and np.mean(q <= 0.02) > 0.5 and np.median(R) < 0.99),
        )
    dmax = 0.0
    for r in rows:
        if r["cell"].startswith("qres_dynoff") and not r["cell"].endswith("_p0.00"):
            p = float(r["cell"].split("_p")[1])
            pA1 = r["joint_AB"][0] + r["joint_AB"][1]
            pred = (1.0 - p) * (pA1**2 + (1.0 - pA1) ** 2) + p
            dmax = max(dmax, abs(pred - r["R_rep_A"]))
    return summ, dmax


def e8_table():
    rows = load("E8")
    out = {}
    for r in rows:
        out.setdefault(r["cell"], []).append(r)
    summ = {}
    for cell, rs in sorted(out.items()):
        S0 = np.array([r["S0"] for r in rs])
        oss = np.array([r["OSS"] for r in rs])
        q = np.array([abs(r["q_QQ"]) for r in rs])
        summ[cell] = dict(
            n=len(rs),
            S0_mean=float(S0.mean()),
            OSS_mean=float(oss.mean()),
            qQQ_absmed=float(np.median(q)),
            corr_OSS_S0=float(np.corrcoef(S0, oss)[0, 1]) if S0.std() > 1e-12 else float("nan"),
        )
    return summ


def e1b_table():
    rows = load("E1b")
    out = {}
    for r in rows:
        out.setdefault(r["cell"], []).append(r["tunable_fraction"])
    return {
        c: dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)), frac_any_hit=float(np.mean(np.array(v) > 0)))
        for c, v in sorted(out.items())
    }


def e1c_table():
    rows = load("E1c")
    q = np.array([abs(r["q_QQ"]) for r in rows])
    oss = np.array([r["OSS"] for r in rows])
    xya = np.array([xy_pair(r) for r in rows])
    sl, _ = np.polyfit(xya[:, 0], xya[:, 1], 1)
    rxy = float(np.corrcoef(xya[:, 0], xya[:, 1])[0, 1])
    cbd = float(np.mean([A.cbd2_lp(r["joint_AB"], r["joint_BA"])["feasible"] for r in rows]))
    return dict(
        n=len(rows),
        qQQ_absmed=float(np.median(q)),
        band=float(np.mean((q <= 0.02) & (oss >= 0.05))),
        OSS_mean=float(oss.mean()),
        xy_slope=float(sl),
        xy_r=rxy,
        cbd2_nonctx=cbd,
        R_rep_med=float(np.median([r["R_rep_A"] for r in rows])),
    )


def grain_table():
    rows = [r for r in load("E6") if r["cell"].startswith("dyn0_n4_")]
    out = {}
    for r in rows:
        cell = r["cell"]
        pu = float(cell.split("_pu")[1].split("_")[0])
        pd = float(cell.split("_pd")[1])
        if abs(pu - pd) > 1e-9:
            continue
        a, b = r["a_rate"], r["b_rate"]
        if np.isnan(a) or np.isnan(b):
            continue
        out.setdefault(cell, []).append((a - b, r["q_QQ"]))
    return {
        cell: dict(
            grain_absmed=float(np.median(np.abs([p[0] for p in pairs]))),
            qQQ_absmed=float(np.median(np.abs([p[1] for p in pairs]))),
        )
        for cell, pairs in sorted(out.items())
    }


def make_figures(e1, e5, e4, e2, e6s, e7s, e8s):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows_e1 = load("E1")
    subs = ["luders_qubit", "lin_classical", "nl_classical", "qreservoir"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, s in zip(axes, subs):
        rs = [r for r in rows_e1 if r["substrate"] == s]
        if s == "lin_classical":
            rs = [r for r in rs if proper(r)]
        q = np.array([q_qq(r) for r in rs])
        oss = np.array([abs(r["q_A"]) + abs(r["q_B"]) for r in rs])
        yj = np.random.default_rng(0).uniform(-0.35, 0.35, size=len(q))
        inb = (np.abs(q) <= 0.02) & (oss >= 0.05)
        colors = np.where(inb, "#2a9d8f", "#e76f51")
        ax.axvspan(-0.02, 0.02, color="#2a9d8f", alpha=0.12, lw=0)
        ax.scatter(q, yj, s=2.5, c=colors, alpha=0.35, lw=0)
        ax.set_title(f"{s}\nin band: {inb.mean():.2f}", fontsize=10)
        ax.set_xlabel("q_QQ")
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
    axes[0].set_ylabel("instances (jitter)")
    fig.suptitle("Distribution of q_QQ by substrate (shaded: |q_QQ| <= 0.02; green: also OSS >= 0.05)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F1_e1_qQQ_distributions.png", dpi=160)
    plt.close(fig)

    anchors = {
        "Clinton-Gore": (-0.0726, 0.0756, "#1d3557"),
        "White-Black": (-0.0025, 0.0215, "#1d3557"),
        "Rose-Jackson": (-0.0777, -0.0737, "#e63946"),
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), sharex=True, sharey=True)
    lim = 0.45
    for ax, s in zip(axes, subs):
        rs = [r for r in rows_e1 if r["substrate"] == s]
        if s == "lin_classical":
            rs = [r for r in rs if proper(r)]
        xya = np.array([xy_pair(r) for r in rs])
        ax.scatter(xya[:, 0], xya[:, 1], s=2.5, alpha=0.3, c="#457b9d", lw=0)
        xx = np.linspace(-lim, lim, 10)
        ax.plot(xx, -xx, "k--", lw=1, label="y = -x")
        sl, ic = np.polyfit(xya[:, 0], xya[:, 1], 1)
        ax.plot(xx, sl * xx + ic, "-", color="#e76f51", lw=1.6, label=f"fit (slope {sl:+.2f})")
        for _, (ax_, ay_, c_) in anchors.items():
            ax.scatter([ax_], [ay_], marker="*", s=140, c=c_, zorder=5)
        ax.set_title(s, fontsize=10)
        ax.set_xlabel("x = Delta p(yy)")
        ax.axhline(0, color="gray", lw=0.4)
        ax.axvline(0, color="gray", lw=0.4)
    axes[0].set_ylabel("y = Delta p(nn)")
    axes[0].legend(loc="lower left", fontsize=7)
    axes[1].legend(loc="upper right", fontsize=7)
    handles = [
        plt.Line2D([], [], marker="*", ls="", c="#1d3557", label="human: back-to-back"),
        plt.Line2D([], [], marker="*", ls="", c="#e63946", label="human: interpolated text"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("Diagonal cell differences (x, y) versus anti-diagonal y = -x", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "F2_xy_anti_diagonal.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, key, title, xlab in [
        (axes[0], "qres_dyn", "quantum reservoir, dynamics ON", "back-action strength p"),
        (axes[1], "qres_nodyn", "quantum reservoir, dynamics OFF", "back-action strength p"),
        (axes[2], "nlc", "classical sigmoid, feedback sweep", "feedback scale v"),
    ]:
        d = e4[key]
        ps = sorted(d)
        med = [d[p][0] for p in ps]
        p75 = [d[p][1] for p in ps]
        ax.plot(ps, med, "o-", ms=3)
        ax.fill_between(ps, med, p75, alpha=0.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlab)
        ax.set_ylabel("median |q_QQ|")
        ax.set_ylim(-0.005, max(0.05, max(p75) * 1.1))
    fig.suptitle("Median |q_QQ| across back-action and feedback sweeps (shaded: median to 75th percentile)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "F3_e4_dials.png", dpi=160)
    plt.close(fig)

    cells = sorted(e5)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = np.arange(len(cells))
    ax.bar(xs - 0.22, [e5[c]["frac_violation"] for c in cells], width=0.4, label="frac |q_QQ| > 0.05")
    ax.bar(xs + 0.22, [e5[c]["band"] for c in cells], width=0.4, label="human-band fraction")
    ax2 = ax.twinx()
    ax2.plot(xs, [e5[c]["OSS_mean"] for c in cells], "k^--", ms=5, lw=1, label="OSS mean (right)")
    ax2.set_ylabel("OSS mean")
    ax.set_xticks(xs)
    ax.set_xticklabels(cells, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.0)
    ax.set_title("Factorial localization across dynamics, measurement update, fading, and feedback", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F4_e5_localization.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cells2 = sorted(e2)
    xs2 = np.arange(len(cells2))
    axes[0].bar(xs2 - 0.2, [e2[c]["lp_feasible_rate"] for c in cells2], width=0.4, label="global section feasible")
    axes[0].bar(xs2 + 0.2, [e2[c]["cbd_feasible_rate"] for c in cells2], width=0.4, label="cyclic-3 CbD noncontextual")
    axes[0].set_xticks(xs2)
    axes[0].set_xticklabels(cells2, rotation=35, ha="right", fontsize=7)
    axes[0].set_ylabel("rate")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Three-question feasibility rates")
    axes[1].bar(xs2, [e2[c]["eps_star_median"] for c in cells2], color="#457b9d")
    axes[1].set_xticks(xs2)
    axes[1].set_xticklabels(cells2, rotation=35, ha="right", fontsize=7)
    axes[1].set_ylabel("median epsilon*")
    axes[1].set_title("Median global-section L1 violation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F5_e2_contextuality.png", dpi=160)
    plt.close(fig)

    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, (n, dyn) in zip(axes, [(1, False), (4, False), (4, True)]):
        M = np.zeros((5, 5))
        for i, pu in enumerate(grid):
            for j, pd in enumerate(grid):
                cell = f"dyn{int(dyn)}_n{n}_pu{pu:.2f}_pd{pd:.2f}"
                M[i, j] = e6s[cell]["qQQ_absmed"] if cell in e6s else np.nan
        im = ax.imshow(M, origin="lower", cmap="viridis", vmin=0, vmax=max(0.05, np.nanmax(M)))
        ax.set_xticks(range(5), [f"{g:g}" for g in grid])
        ax.set_yticks(range(5), [f"{g:g}" for g in grid])
        ax.set_xlabel("p_down (back-action on 'no')")
        ax.set_ylabel("p_up (back-action on 'yes')")
        ax.set_title(f"dynamics {'ON' if dyn else 'OFF'}, n={n}", fontsize=10)
        fig.colorbar(im, ax=ax, label="median |q_QQ|")
    fig.suptitle("Asymmetry plane of outcome-conditioned back-action", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_DIR / "F6_e6_asymmetry_plane.png", dpi=160)
    plt.close(fig)

    p_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    for name, color in [
        ("qres_dynoff", "#2a9d8f"),
        ("qres_dynon", "#e76f51"),
        ("nl_classical", "#457b9d"),
        ("lin2_classical", "#8d5a97"),
    ]:
        xs3 = [p for p in p_grid if f"{name}_p{p:.2f}" in e7s]
        ys3 = [e7s[f"{name}_p{p:.2f}"]["R_med"] for p in xs3]
        ax.plot(xs3, ys3, "o-", ms=4, color=color, label=name)
    ax.set_xlabel("back-action strength p")
    ax.set_ylabel("median replicability R")
    ax.set_title("Response replicability vs back-action strength", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    for name, mk in [
        ("qres_dynoff", "o"),
        ("qres_dynon", "s"),
        ("nl_classical", "^"),
        ("lin2_classical", "v"),
        ("luders_qubit", "*"),
    ]:
        xs3, ys3 = [], []
        for p in p_grid:
            cell = f"{name}_p{p:.2f}"
            if cell in e7s:
                m = e7s[cell]
                ok = m["triad"]
                ax.scatter(
                    m["R_med"],
                    m["OSS_mean"],
                    marker=mk,
                    c="#2a9d8f" if ok else "#b8b8b8",
                    s=70 if ok else 25,
                    edgecolors="k",
                    linewidths=0.4,
                    zorder=3,
                )
                xs3.append(m["R_med"])
                ys3.append(m["OSS_mean"])
        if xs3:
            ax.plot(xs3, ys3, "-", color="#999999", lw=0.7, zorder=1)
    ax.set_xlabel("median R (replicability)")
    ax.set_ylabel("OSS mean")
    ax.set_title("Joint regime: OSS >= 0.05, |q_QQ| <= 0.02, R < 0.99 (green)", fontsize=10)
    fig.suptitle("Order effects, response replicability, and QQ satisfaction", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_DIR / "F7_e7_replicability_triad.png", dpi=160)
    plt.close(fig)

    rows_e8 = load("E8")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for mode, c in {"product": "#e76f51", "maxpair": "#2a9d8f", "haar": "#457b9d"}.items():
        rs = [r for r in rows_e8 if r["cell"] == mode]
        ax.scatter(
            [r["S0"] for r in rs],
            [r["OSS"] for r in rs],
            s=4,
            alpha=0.3,
            c=c,
            lw=0,
            label=f"{mode} (S0={e8s[mode]['S0_mean']:.2f})",
        )
    ax.set_xlabel("initial entanglement entropy S0 (nats)")
    ax.set_ylabel("OSS")
    ax.set_title("Order sensitivity vs initial entanglement entropy", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F8_e8_entanglement.png", dpi=160)
    plt.close(fig)


def write_summary(e1, e5, e4, e2, e6s, r_lin, e7s, dmax_r, e8s, e1bs, e1cs, grain):
    lines = ["=== E1 Substrate Comparison ==="]
    for s, m in e1.items():
        lines.append(
            f"{s:15s} n={m['n']:4d} med|q_QQ|={m['qQQ_absmedian']:.3f} band={m['frac_band']:.3f} "
            f"OSS_mean={m['OSS_mean']:.3f} OSS_max={m['OSS_max']:.3f} human_band={m['band']:.3f} "
            f"xy_slope={m['xy_slope']:+.3f} xy_r={m['xy_r']:+.3f} "
            f"rank2_ctx={1.0 - m['cbd2_noncontextual_rate']:.3f} "
            f"above_ceil={m['frac_above_ceil']:.4f} ellipse_infeas={m['frac_ellipse_infeas']:.4f}"
        )

    lines.append("\n=== E5 Localization ===")
    for c, m in e5.items():
        lines.append(f"{c:15s} n={m['n']:4d} med|q_QQ|={m['qQQ_absmedian']:.3f} OSS_mean={m['OSS_mean']:.3f} band={m['band']:.3f}")

    lines.append("\n=== E4 Sweeps ===")
    for key in ("qres_dyn", "qres_nodyn", "nlc"):
        lines.append(f"[{key}] " + ", ".join(f"{p:.2f}:{v[0]:.4f}" for p, v in e4[key].items()))

    lines.append("\n=== E2 3-Question Batteries ===")
    for c, m in e2.items():
        lines.append(
            f"{c:28s} lp_feas={m['lp_feasible_rate']:.3f} eps_med={m['eps_star_median']:.3f} cbd_nonctx={m['cbd_feasible_rate']:.3f}"
        )

    lines.append(f"\n=== E6 Asymmetry Plane (dyn0_n1 linearity r = {r_lin:.4f}) ===")
    for cell, m in e6s.items():
        lines.append(f"{cell:24s} med|q_QQ|={m['qQQ_absmed']:.4f} iqr75={m['qQQ_iqr75']:.4f}")

    lines.append("\n=== Grain Analysis (dyn0_n4 symmetric) ===")
    for cell, m in grain.items():
        lines.append(f"{cell:24s} med|grain|={m['grain_absmed']:.4f} med|q_QQ|={m['qQQ_absmed']:.5f}")

    lines.append(f"\n=== E7 Replicability Triad (closed-form maxdiff = {dmax_r:.2e}) ===")
    for cell, m in e7s.items():
        lines.append(f"{cell:22s} R_med={m['R_med']:.3f} OSS_mean={m['OSS_mean']:.3f} med|q_QQ|={m['qQQ_absmed']:.4f} triad={m['triad']}")

    lines.append("\n=== E8 Entanglement ===")
    for cell, m in e8s.items():
        lines.append(f"{cell:10s} S0_mean={m['S0_mean']:.3f} OSS_mean={m['OSS_mean']:.3f} corr(OSS,S0)={m['corr_OSS_S0']:+.3f} med|q_QQ|={m['qQQ_absmed']:.4f}")

    lines.append("\n=== E1b Readout-Only Retuning ===")
    for c, m in e1bs.items():
        lines.append(f"{c:22s} mean={m['mean']:.3f} median={m['median']:.3f} any_hit={m['frac_any_hit']:.3f}")

    lines.append("\n=== E1c Constrained Linear Ensemble ===")
    lines.append(
        f"n={e1cs['n']} med|q_QQ|={e1cs['qQQ_absmed']:.4f} band={e1cs['band']:.3f} "
        f"OSS_mean={e1cs['OSS_mean']:.3f} xy_slope={e1cs['xy_slope']:+.3f} xy_r={e1cs['xy_r']:+.3f} "
        f"rank2_nonctx={e1cs['cbd2_nonctx']:.3f} R_med={e1cs['R_rep_med']:.3f}"
    )

    lines.append("\n=== E12 Adversarial Optimization ===")
    for r in load("E12"):
        lines.append(f"{r['instance_id']:30s} max_OSS={r['max_OSS']:.4f}")

    text = "\n".join(lines) + "\n"
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    e1 = e1_table()
    e5 = e5_table()
    e4 = e4_curves()
    e2 = e2_table()
    e6s, r_lin = e6_table()
    e7s, dmax_r = e7_table()
    e8s = e8_table()
    e1bs = e1b_table()
    e1cs = e1c_table()
    grain = grain_table()
    write_summary(e1, e5, e4, e2, e6s, r_lin, e7s, dmax_r, e8s, e1bs, e1cs, grain)
    make_figures(e1, e5, e4, e2, e6s, e7s, e8s)
    print("Figures saved to results/figures/")
