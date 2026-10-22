"""Figures + summary from envguide/scores.csv. Re-run any time; it uses whatever has been scored so far.

  python envguide/figures.py [--base unguided] [--guided guided]

Writes envguide/figures/*.png and envguide/figures/summary.md:
  1_guided_vs_unguided.png  per-target TM-score, Boltz top-confidence model and best of N (option 2)
  2_cc_vs_tm.png            envelope CC vs TM-score for every model (does envelope fit track accuracy?)
  3_selection.png           option 1: pick by Boltz confidence vs by envelope CC vs random vs oracle
  4_guidance_trace.png      envelope CC of x0 during guided sampling
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
import sys; sys.path.insert(0, HERE)
from run_boltz import RUNS  # noqa: E402
C_BASE, C_GUIDE, C_CC, C_GREY = "#4C78A8", "#E45756", "#54A24B", "#9D9D9D"


def per_target(df, run):
    d = df[df.run == run]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("emdb")
    top = d.loc[g.confidence.idxmax()].set_index("emdb")
    return pd.DataFrame({
        "n": g.size(),
        "tm_top_conf": top.tm_ref,
        "tm_best": g.tm_ref.max(),
        "tm_mean": g.tm_ref.mean(),
        "cc_top_conf": top.env_cc,
        "tm_top_cc": d.loc[g.env_cc.idxmax()].set_index("emdb").tm_ref,
        "year": g.year.first(),
    })


def fig_option2(df, base, guided, lines):
    a, b = per_target(df, base), per_target(df, guided)
    common = a.index.intersection(b.index)
    if len(common) == 0:
        lines.append(f"_No targets with both `{base}` and `{guided}` yet._\n")
        return
    a, b = a.loc[common], b.loc[common]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(common))
    for ax, col, title in [(axs[0], "tm_top_conf", "Top-confidence model"), (axs[1], "tm_best", f"Best of N")]:
        ax.bar(x - 0.2, a[col], 0.4, color=C_BASE, label=base)
        ax.bar(x + 0.2, b[col], 0.4, color=C_GUIDE, label=guided)
        for i, e in enumerate(common):
            dlt = b[col][e] - a[col][e]
            ax.text(i, max(a[col][e], b[col][e]) + 0.01, f"{dlt:+.2f}", ha="center", fontsize=8,
                    color=C_GUIDE if dlt > 0 else C_BASE)
        ax.set_xticks(x); ax.set_xticklabels(common, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("TM-score vs deposited"); ax.set_title(title)
        ax.axhline(0.5, color=C_GREY, lw=0.8, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Option 2: envelope-guided diffusion vs plain Boltz-2 ({len(common)} maps)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_guided_vs_unguided.png"), dpi=150); plt.close(fig)

    d_top, d_best = b.tm_top_conf - a.tm_top_conf, b.tm_best - a.tm_best
    d_cc = b.cc_top_conf - a.cc_top_conf
    lines += [f"## Option 2: `{guided}` vs `{base}` ({len(common)} maps)\n",
              "| map | year | TM top-conf base | guided | delta | TM best-of-N base | guided | delta | env CC base | guided |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for e in common:
        lines.append(f"| {e} | {a.year[e]} | {a.tm_top_conf[e]:.3f} | {b.tm_top_conf[e]:.3f} | {d_top[e]:+.3f} | "
                     f"{a.tm_best[e]:.3f} | {b.tm_best[e]:.3f} | {d_best[e]:+.3f} | {a.cc_top_conf[e]:.3f} | {b.cc_top_conf[e]:.3f} |")
    lines += ["", f"- Top-confidence TM-score: mean delta {d_top.mean():+.3f}, median {d_top.median():+.3f}; "
              f"improved {int((d_top > 0.01).sum())}, worse {int((d_top < -0.01).sum())}, "
              f"same (±0.01) {int((d_top.abs() <= 0.01).sum())}.",
              f"- Best-of-N TM-score: mean delta {d_best.mean():+.3f}; improved {int((d_best > 0.01).sum())}, "
              f"worse {int((d_best < -0.01).sum())}.",
              f"- Envelope CC of the top model: mean delta {d_cc.mean():+.3f} (guidance should raise this; "
              "if CC rises but TM does not, the envelope is being fitted without getting the structure right).", ""]


def fig_cc_vs_tm(df, lines):
    runs = sorted(df.run.unique())
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    cols = {r: c for r, c in zip(runs, [C_BASE, C_GUIDE, C_CC, "#B279A2", "#FF9DA6"])}
    for r in runs:
        d = df[df.run == r]
        ax.scatter(d.env_cc, d.tm_ref, s=14, alpha=0.7, color=cols[r], label=r)
    ax.set_xlabel("Envelope CC (global rigid fit)"); ax.set_ylabel("TM-score vs deposited")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    # within-target rank correlation is what matters for selection
    rho = []
    for (r, e), d in df.groupby(["run", "emdb"]):
        if len(d) >= 4 and d.tm_ref.std() > 0:
            rho.append(d.env_cc.rank().corr(d.tm_ref.rank()))
    t = f"Within-map Spearman(CC, TM): median {np.median(rho):+.2f} over {len(rho)} sets" if rho else ""
    ax.set_title(t, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_cc_vs_tm.png"), dpi=150); plt.close(fig)
    if rho:
        lines += ["## Does envelope fit track accuracy?\n", f"- {t}.", ""]


def fig_selection(df, run, lines):
    d = df[df.run == run]
    if d.empty:
        return
    rows = []
    for e, g in d.groupby("emdb"):
        if len(g) < 2:
            continue
        rows.append(dict(emdb=e, n=len(g), confidence=g.loc[g.confidence.idxmax()].tm_ref,
                         env_cc=g.loc[g.env_cc.idxmax()].tm_ref,
                         combined=g.loc[(g.confidence.rank() + g.env_cc.rank()).idxmax()].tm_ref,
                         random=g.tm_ref.mean(), oracle=g.tm_ref.max()))
    if not rows:
        return
    s = pd.DataFrame(rows).set_index("emdb")
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(s) + 3), 4.6))
    x = np.arange(len(s)); wdt = 0.16
    for i, (col, c) in enumerate([("random", C_GREY), ("confidence", C_BASE), ("env_cc", C_CC),
                                  ("combined", "#B279A2"), ("oracle", "#333333")]):
        ax.bar(x + (i - 2) * wdt, s[col], wdt, color=c, label={"random": "random (mean)", "confidence": "Boltz confidence",
               "env_cc": "envelope CC", "combined": "conf + CC ranks", "oracle": "oracle (best)"}[col])
    ax.set_xticks(x); ax.set_xticklabels(s.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("TM-score of selected model")
    ax.set_title(f"Option 1: pick one of N samples ({run}, N={int(s.n.median())})")
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_selection.png"), dpi=150); plt.close(fig)
    dl = s.env_cc - s.confidence
    lines += [f"## Option 1: selection among samples of `{run}` ({len(s)} maps, N≈{int(s.n.median())})\n",
              "| map | random | by confidence | by envelope CC | conf+CC | oracle |", "|---|---|---|---|---|---|"]
    for e in s.index:
        lines.append(f"| {e} | {s.random[e]:.3f} | {s.confidence[e]:.3f} | {s.env_cc[e]:.3f} | {s.combined[e]:.3f} | {s.oracle[e]:.3f} |")
    lines += ["", f"- Envelope-CC pick minus confidence pick: mean {dl.mean():+.3f}; better on {int((dl > 0.01).sum())}, "
              f"worse on {int((dl < -0.01).sum())} maps. Headroom (oracle - confidence): mean {(s.oracle - s.confidence).mean():.3f}.", ""]


def fig_trace(run, lines):
    logs = sorted(glob.glob(os.path.join(RUNS, run, "EMD-*", "guide_log.json")))
    logs = [(p, json.load(open(p))) for p in logs]
    logs = [(p, j) for p, j in logs if j["log"]]
    if not logs:
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for p, j in logs:
        st = [e["step"] for e in j["log"]]
        cc = [np.mean(e["cc_after"]) for e in j["log"]]
        ax.plot(st, cc, lw=1, label=p.split(os.sep)[-2])
    ax.set_xlabel("diffusion step"); ax.set_ylabel("envelope CC of x0 (mean over samples)")
    ax.set_title(f"Guidance trace ({run})", fontsize=10); ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "4_guidance_trace.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unguided")
    ap.add_argument("--guided", default="guided")
    ap.add_argument("--select-run", default=None, help="run used for option-1 selection (default: largest-N run)")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(os.path.join(HERE, "scores.csv"))
    lines = ["# Envelope-guided prediction: results so far\n",
             f"{len(df)} models scored across {df.emdb.nunique()} maps; runs: "
             + ", ".join(f"`{r}` ({(df.run == r).sum()})" for r in sorted(df.run.unique())) + ".\n",
             "TM-score is against the deposited fitted model (normalised by its length; chain mapping by US-align). "
             "Most deposited models predate Boltz-2's training cutoff, so absolute scores are optimistic; "
             "differences between modes are the point.\n"]
    fig_option2(df, a.base, a.guided, lines)
    sel = a.select_run or df.groupby("run").size().idxmax()
    fig_selection(df, sel, lines)
    fig_cc_vs_tm(df, lines)
    fig_trace(a.guided, lines)
    open(os.path.join(FIG, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
