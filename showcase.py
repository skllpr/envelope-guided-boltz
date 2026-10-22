import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json, os
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # modal_results/ = what `modal run modal_pilot.py::fetch` downloads from the volume
    RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modal_results")
    FIGS = os.path.join(RES, "showcase_figures")  # every figure below is also saved here as PNG
    C_PLAIN, C_GUIDED, C_ADAPT, C_DEP, C_ENV, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df", "#52514e"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ae",
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.size": 9})
    return C_ADAPT, C_DEP, C_ENV, C_GUIDED, C_PLAIN, FIGS, RES, json, mo, np, os, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Does a negative-stain envelope make Boltz-2 more accurate?

    Each map was predicted twice with Boltz-2 from sequence alone: **plain**, and **envelope-guided** (every
    diffusion step nudged toward the ~20 Å negative-stain envelope). Accuracy is the TM-score against the
    deposited model (0–1, higher is better; above 0.5 means the same overall fold/arrangement). Each mode's
    representative model is the one Boltz is most confident in, the one you would pick without knowing the answer.
    """)
    return


@app.cell
def _(RES, json, mo, os, pd):
    _summary = os.path.join(RES, "analysis", "summary.json")
    mo.stop(not os.path.exists(_summary), mo.callout(mo.md(
        f"No analysis found at `{_summary}`. Run `modal run modal_pilot.py::fetch` (after the pilot and "
        "analysis have finished) from this folder."), kind="warn"))
    S = json.load(open(_summary))
    BASE, GUIDED, ADAPT = S["base"], S["guided"], S.get("adaptive")  # ADAPT: confidence-aware run, if analysed
    maps = {m["emdb"]: m for m in S["maps"]}
    scores = pd.read_csv(os.path.join(RES, "scores.csv"))
    return ADAPT, BASE, GUIDED, maps, scores


@app.cell
def _(ADAPT, BASE, C_ADAPT, C_DEP, C_ENV, C_GUIDED, C_PLAIN, GUIDED, RES, maps, np, os, plt, scores):
    # Figure builders, shared by the display cells and the save-all cell.
    MODE_STYLE = {BASE: (C_PLAIN, "o", "plain"), GUIDED: (C_GUIDED, "^", "envelope-guided")}
    ADAPT_STYLE = (C_ADAPT, "s", "confidence-aware guidance")

    def fig_overview():
        """Every sample (small) and the top-confidence model (large) per map; arrow plain -> guided;
        square = confidence-aware guidance (guidance scaled down when plain Boltz is confident)."""
        order = sorted(maps, key=lambda e: maps[e]["delta_tm_top_conf"])
        has_adapt = ADAPT is not None and any(ADAPT in maps[e] for e in order)
        fig, a = plt.subplots(figsize=(8.6 if has_adapt else 8, 0.8 * len(order) + 1.5))
        for i, e in enumerate(order):
            for k, (mode, (col, mk, lab)) in enumerate(MODE_STYLE.items()):
                d = scores[(scores.emdb == e) & (scores.run == mode)]
                y = i + (0.12 if k else -0.12)
                a.scatter(d.tm_ref, [y] * len(d), s=18, marker=mk, color=col, alpha=0.35, linewidths=0, zorder=2)
                a.scatter([maps[e][mode]["tm_top_conf"]], [y], s=90, marker=mk, color=col, edgecolors="white",
                          linewidths=2, zorder=4, label=f"{lab}: top-confidence model" if i == 0 else None)
            x0, x1 = maps[e][BASE]["tm_top_conf"], maps[e][GUIDED]["tm_top_conf"]
            a.annotate("", xy=(x1, i + 0.12), xytext=(x0, i - 0.12), zorder=3,
                       arrowprops=dict(arrowstyle="->", color="#8a8984", lw=1.2, shrinkA=6, shrinkB=6))
            a.text(1.03, i, f"{x1 - x0:+.2f}", va="center", fontsize=10, color="#0b0b0b",
                   fontweight="bold" if abs(x1 - x0) > 0.02 else "normal")
            if has_adapt and ADAPT in maps[e]:
                xa = maps[e][ADAPT]["tm_top_conf"]
                a.scatter([xa], [i + 0.34], s=70, marker=ADAPT_STYLE[1], color=C_ADAPT, edgecolors="white",
                          linewidths=2, zorder=5, label=f"{ADAPT_STYLE[2]}: top-confidence model" if i == 0 else None)
                a.text(1.16, i, f"{xa - x0:+.2f}", va="center", fontsize=10, color="#0b0b0b",
                       fontweight="bold" if abs(xa - x0) > 0.02 else "normal")
        a.set_yticks(range(len(order)))
        a.set_yticklabels([f"{e}\n{maps[e]['pdb']}, {maps[e]['chains']} chains" for e in order], fontsize=8)
        a.set_xlim(0, 1.0); a.set_ylim(-0.6, len(order) - 0.4 + (0.2 if has_adapt else 0))
        a.axvline(0.5, color="#b5b4ae", lw=0.8, ls=":")
        a.set_xlabel("TM-score vs deposited model (higher is better)")
        top = len(order) - 0.45 + (0.2 if has_adapt else 0)
        a.text(1.03, top, "Δ guided", fontsize=8, color="#52514e")
        if has_adapt:
            a.text(1.16, top, "Δ conf-aware", fontsize=8, color="#52514e")
        a.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3 if has_adapt else 2)
        a.set_title("Envelope guidance vs plain Boltz-2 (small marks: every sample)", loc="left", fontsize=10,
                    pad=12)
        fig.tight_layout()
        return fig

    def fig_adaptive():
        """Plain vs confidence-aware guidance only: one arrow per map from the plain top model to the
        confidence-aware top model. Right = more accurate; solid arrow = change > 0.05 TM, dashed = smaller."""
        if ADAPT is None:
            return None
        order = sorted((e for e in maps if ADAPT in maps[e]), key=lambda e: maps[e][ADAPT]["tm_top_conf"]
                       - maps[e][BASE]["tm_top_conf"])
        fig, a = plt.subplots(figsize=(9, 0.62 * len(order) + 1.6))
        for i, e in enumerate(order):
            x0, x1 = maps[e][BASE]["tm_top_conf"], maps[e][ADAPT]["tm_top_conf"]
            d = x1 - x0
            big = abs(d) > 0.02
            long = abs(d) > 0.05  # long enough for a visible arrow; shorter: offset so neither marker hides
            y0, y1 = (i, i) if long else (i - 0.12, i + 0.12)
            if long:
                a.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=3, arrowprops=dict(
                    arrowstyle="-|>,head_width=0.35,head_length=0.7", color="#3d3d3a", lw=1.8, shrinkA=7, shrinkB=7))
            else:
                a.plot([x0, x1], [y0, y1], color="#b5b4ae", lw=1.2, ls="--", zorder=3)
            a.scatter([x0], [y0], s=110, marker="o", color=C_PLAIN, edgecolors="white", linewidths=2, zorder=4,
                      label="plain Boltz-2" if i == 0 else None)
            a.scatter([x1], [y1], s=100, marker="s", color=C_ADAPT, edgecolors="white", linewidths=2, zorder=5,
                      label="confidence-aware envelope guidance" if i == 0 else None)
            word = "improved" if d > 0.02 else "worse" if d < -0.02 else "unchanged"
            glyph = "▲" if d > 0.02 else "▼" if d < -0.02 else "="
            a.text(1.03, i, f"{glyph} {d:+.2f}  {word}", va="center", fontsize=10, color="#0b0b0b",
                   fontweight="bold" if big else "normal")
            p = maps[e].get("adaptive_plan")
            if p:
                a.text(1.03, i - 0.3, f"plain ipTM {p['plain_iptm']:.2f} → guidance {p['strength']:.0%}",
                       va="center", fontsize=7.5, color="#52514e")
        a.set_yticks(range(len(order)))
        a.set_yticklabels([f"{e}\n{maps[e]['pdb']}, {maps[e]['chains']} chains" for e in order], fontsize=8)
        a.set_xlim(0, 1.0); a.set_ylim(-0.7, len(order) - 0.4)
        a.axvline(0.5, color="#b5b4ae", lw=0.8, ls=":")
        a.set_xlabel("TM-score vs deposited model (higher is better)")
        dm = np.mean([maps[e][ADAPT]["tm_top_conf"] - maps[e][BASE]["tm_top_conf"] for e in order])
        a.set_title(f"Confidence-aware envelope guidance vs plain Boltz-2 ({len(order)} maps, mean Δ TM {dm:+.2f})",
                    loc="left", fontsize=10, pad=12)
        a.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2)
        fig.tight_layout()
        return fig

    def load_env(e):
        z = np.load(os.path.join(RES, "analysis", e, "envelope.npz"))
        return z, [z["ax_x"], z["ax_y"], z["ax_z"]], z["env"] > 0  # standardised + skewed: >0 = the particle

    def inside(P, ax, mask):  # fraction of CA atoms inside the 3D envelope
        i = [np.clip(np.round((P[:, k] - ax[k][0]) / (ax[k][1] - ax[k][0])).astype(int), 0, len(ax[k]) - 1)
             for k in range(3)]
        return mask[i[0], i[1], i[2]].mean()

    def fig_envelope(e):
        """The envelope down x, y and z with each mode's top model (colour) over the deposited model (grey)."""
        z, ax, mask = load_env(e)
        views = [(0, 1, 2, "x", "y"), (0, 2, 1, "x", "z"), (1, 2, 0, "y", "z")]  # (horizontal, vertical, projected)
        fig, axs = plt.subplots(2, 3, figsize=(11, 7.2))
        for r, mode in enumerate([BASE, GUIDED]):
            col, _, lab = MODE_STYLE[mode]
            for c, (h, v, p, hl, vl) in enumerate(views):
                a = axs[r, c]
                proj = mask.any(axis=p).T.astype(float)  # rows = vertical axis, cols = horizontal
                a.contourf(ax[h], ax[v], proj, levels=[0.5, 1.5], colors=[C_ENV])
                a.contour(ax[h], ax[v], proj, levels=[0.5], colors=["#b5b4ae"], linewidths=1)
                dep, mod = z["deposited"], z[mode]
                a.scatter(dep[:, h], dep[:, v], s=3, color=C_DEP, alpha=0.35, linewidths=0, label="deposited")
                a.scatter(mod[:, h], mod[:, v], s=4, color=col, alpha=0.8, linewidths=0, label=lab)
                a.set_aspect("equal"); a.set_xlabel(f"{hl} (Å)"); a.set_ylabel(f"{vl} (Å)")
                if c == 0:
                    a.set_title(f"{lab.capitalize()}: TM {maps[e][mode]['tm_top_conf']:.2f}\n"
                                f"{inside(mod, ax, mask):.0%} inside envelope "
                                f"(deposited {inside(dep, ax, mask):.0%})", loc="left", fontsize=9)
                if c == 2:
                    a.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
                             markerscale=3)
        fig.suptitle(f"{e} (PDB {maps[e]['pdb']}): models placed in the negative-stain envelope (light grey)",
                     x=0.01, ha="left")
        fig.tight_layout()
        return fig

    def traces(a, P, chains, h, v, **kw):
        """CA trace per chain; breaks where consecutive CAs are > 4.5 Å apart (gaps)."""
        for ch in dict.fromkeys(chains):
            Q = P[chains == ch]
            seg = np.where(np.linalg.norm(np.diff(Q, axis=0), axis=1) > 4.5)[0] + 1
            for s in np.split(Q, seg):
                a.plot(s[:, h], s[:, v], **kw)

    def fig_structures(e):
        """Backbone traces of each mode's top model over the deposited model, top and side view of the complex."""
        z, _, _ = load_env(e)
        dep = z["deposited"]
        mu = dep.mean(0)
        w, V = np.linalg.eigh(np.cov((dep - mu).T))
        V = V[:, ::-1]  # principal axes of the deposited complex, longest first
        proj = lambda P: (P - mu) @ V
        D = proj(dep)
        fig, axs = plt.subplots(2, 2, figsize=(10, 9.4))
        for r, mode in enumerate([BASE, GUIDED]):
            col, _, lab = MODE_STYLE[mode]
            M = proj(z[mode])
            s = maps[e][mode]
            for c, (h, v, view) in enumerate([(0, 1, "top view"), (0, 2, "side view")]):
                a = axs[r, c]
                traces(a, D, z["deposited_chain"], h, v, color=C_DEP, lw=1.6, alpha=0.30)
                traces(a, M, z[f"{mode}_chain"], h, v, color=col, lw=1.0, alpha=0.95)
                a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)
                a.set_title(f"{lab.capitalize()}: TM {s['tm_top_conf']:.2f}, RMSD {s['rmsd_top_conf']:.1f} Å "
                            f"({view})" if c == 0 else view, loc="left", fontsize=10)
        lim = np.abs(np.concatenate([D] + [proj(z[m]) for m in (BASE, GUIDED)])).max() * 1.05
        for a in axs.flat:
            a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        fig.legend(handles=[plt.Line2D([], [], color=C_DEP, lw=2, alpha=0.4, label="deposited"),
                            plt.Line2D([], [], color=C_PLAIN, lw=2, label="plain Boltz-2"),
                            plt.Line2D([], [], color=C_GUIDED, lw=2, label="envelope-guided")],
                   frameon=False, loc="upper left", bbox_to_anchor=(0.005, 0.97), ncol=3, fontsize=9)
        fig.suptitle(f"{e} (PDB {maps[e]['pdb']}, {maps[e]['chains']} chains): top-confidence models "
                     "superposed on the deposited structure", x=0.01, ha="left", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return fig

    def fig_samples(e):
        """Every sample of one map: does guidance raise envelope fit, and does accuracy follow?"""
        fig, a = plt.subplots(figsize=(6, 4.2))
        for mode, (col, mk, lab) in MODE_STYLE.items():
            d = scores[(scores.emdb == e) & (scores.run == mode)]
            a.scatter(d.env_cc, d.tm_ref, s=60, marker=mk, color=col, edgecolors="white", linewidths=2,
                      label=f"{lab} (n={len(d)})", zorder=3)
        a.set_xlabel("Envelope fit (CC)"); a.set_ylabel("TM-score vs deposited")
        a.axhline(0.5, color="#b5b4ae", lw=0.8, ls=":")
        a.legend(frameon=False, fontsize=8)
        a.set_title(f"{e}: every sample", loc="left", fontsize=10)
        fig.tight_layout()
        return fig

    return fig_adaptive, fig_envelope, fig_overview, fig_samples, fig_structures


@app.cell
def _(BASE, GUIDED, fig_overview, maps, mo, pd):
    _rows = []
    for _e, _m in maps.items():
        _rows.append({"map": _e, "PDB": _m["pdb"], "residues": _m["tokens"], "chains": _m["chains"], "year": _m["year"],
                      "plain TM": _m[BASE]["tm_top_conf"], "guided TM": _m[GUIDED]["tm_top_conf"],
                      "Δ TM": _m["delta_tm_top_conf"], "plain best-of-N": _m[BASE]["tm_best"],
                      "guided best-of-N": _m[GUIDED]["tm_best"],
                      "plain vs guided TM": _m["cross_tm"],
                      "plain spread (min pairwise TM)": _m[BASE]["within_tm_min"]})
    overview = pd.DataFrame(_rows).round(3)

    # Showcase: the map where guidance helped most; if it helped nowhere, the map where the guided answer is most
    # different from the plain one relative to the plain samples' own spread (a different conformation).
    _helped = [m for m in maps.values() if m["delta_tm_top_conf"] > 0.02]
    if _helped:
        showcase = max(_helped, key=lambda m: m["delta_tm_top_conf"])["emdb"]
        _why = "largest accuracy gain from the envelope"
    else:
        showcase = min(maps.values(), key=lambda m: m["cross_tm"] - (m[BASE]["within_tm_min"] or 1.0))["emdb"]
        _why = "no map improved by more than 0.02 TM, so this is the map where guidance changed the answer most"
    mo.vstack([mo.md(f"## All maps ({len(maps)})"), fig_overview(), mo.ui.table(overview, selection=None),
               mo.md(f"**Showcase pick: {showcase}** ({_why}). Pick another map below.")])
    return (showcase,)


@app.cell
def _(fig_adaptive, mo):
    _f = fig_adaptive()
    mo.vstack([mo.md("## Confidence-aware guidance vs plain Boltz-2"), _f]) if _f is not None else mo.md("")
    return


@app.cell
def _(ADAPT, BASE, GUIDED, maps, mo, pd):
    # How should one use the envelope? Each strategy's final model per map, scored against the deposited model.
    # The ipTM / envelope-CC picks choose between the plain and guided top-confidence models without the answer.
    _rows = []
    for _e, _m in sorted(maps.items()):
        _b, _g = _m[BASE], _m[GUIDED]
        _r = {"map": _e, "plain": _b["tm_top_conf"], "always guide": _g["tm_top_conf"]}
        if ADAPT and ADAPT in _m:
            _p = _m.get("adaptive_plan", {})
            _r["confidence-aware"] = _m[ADAPT]["tm_top_conf"]
            _r["plain ipTM → strength"] = (f"{_p['plain_iptm']:.2f} → {_p['strength']:.2f}" if _p else "")
        _r["pick by higher ipTM"] = (_g if _g["iptm_top"] > _b["iptm_top"] else _b)["tm_top_conf"]
        _r["pick by envelope fit"] = (_g if _g["cc_top_conf"] > _b["cc_top_conf"] else _b)["tm_top_conf"]
        _rows.append(_r)
    strategies = pd.DataFrame(_rows)
    _num = strategies.select_dtypes("number")
    _mean = _num.mean().round(3)
    _txt = ", ".join(f"{k} **{v:.2f}**" for k, v in _mean.items())
    mo.vstack([
        mo.md("## Strategies: how to use the envelope\n"
              "TM-score of each strategy's final model. *Confidence-aware*: guidance strength set by the plain run's "
              "ipTM (full below 0.5, off above 0.8). *Pick by higher ipTM / envelope fit*: choose between the plain and "
              "always-guided models by that score."),
        mo.ui.table(strategies.round(3), selection=None),
        mo.md(f"Mean TM-score over {len(strategies)} maps: {_txt}."),
        mo.callout(mo.md(
            "Guided models get **lower ipTM on every map**, even where they are far more accurate, so ipTM cannot "
            "judge a guided structure; it is only used here to decide whether to guide at all. The ipTM cut-offs "
            "are the usual AlphaFold-Multimer bands, but only one map in this pilot is confident, so the rule needs "
            "more maps before it can be called validated."), kind="info"),
    ])
    return


@app.cell
def _(maps, mo, showcase):
    pick = mo.ui.dropdown(options=sorted(maps), value=showcase, label="Map")
    pick
    return (pick,)


@app.cell
def _(ADAPT, BASE, GUIDED, maps, mo, pick):
    m = maps[pick.value]
    b, g = m[BASE], m[GUIDED]
    _d = m["delta_tm_top_conf"]
    _extra = []
    if ADAPT and ADAPT in m:
        _p = m.get("adaptive_plan", {})
        _extra = [mo.stat(f"{m[ADAPT]['tm_top_conf']:.2f}", label="Confidence-aware TM-score",
                          caption=(f"plain ipTM {_p['plain_iptm']:.2f} → strength {_p['strength']:.2f}" if _p else ""))]
    mo.hstack([
        mo.stat(f"{b['tm_top_conf']:.2f}", label="Plain TM-score", caption=f"RMSD {b['rmsd_top_conf']:.1f} Å"),
        mo.stat(f"{g['tm_top_conf']:.2f}", label="Guided TM-score", caption=f"RMSD {g['rmsd_top_conf']:.1f} Å",
                direction="increase" if _d > 0 else "decrease" if _d < 0 else None),
        mo.stat(f"{_d:+.2f}", label="Change from the envelope", caption="guided − plain, top-confidence models"),
        mo.stat(f"{b['cc_top_conf']:.2f} → {g['cc_top_conf']:.2f}", label="Envelope fit (CC)",
                caption=f"deposited model: {m['deposited_env_cc']:.2f}"),
    ] + _extra, widths="equal")
    return b, g, m


@app.cell
def _(b, g, m, mo):
    _d, _cross = m["delta_tm_top_conf"], m["cross_tm"]
    _spread = min(x for x in [b["within_tm_min"], g["within_tm_min"]] if x is not None) \
        if any(x is not None for x in [b["within_tm_min"], g["within_tm_min"]]) else None
    if _d > 0.02:
        _acc = f"The envelope **improved** accuracy: TM {b['tm_top_conf']:.2f} → {g['tm_top_conf']:.2f} ({_d:+.2f})."
    elif _d < -0.02:
        _acc = f"The envelope **reduced** accuracy here: TM {b['tm_top_conf']:.2f} → {g['tm_top_conf']:.2f} ({_d:+.2f})."
    else:
        _acc = f"Accuracy was **about the same** (TM {b['tm_top_conf']:.2f} vs {g['tm_top_conf']:.2f})."
    if _spread is None:
        _conf = f"Plain vs guided top models: TM {_cross:.2f} to each other (only one sample per mode, so no spread to compare)."
    elif _cross < _spread - 0.02:
        _conf = (f"The guided answer is a **different arrangement**: the two top models agree only to TM {_cross:.2f}, "
                 f"while samples within a mode agree to at least {_spread:.2f}. That difference is beyond sampling noise.")
    else:
        _conf = (f"The guided and plain answers are **similar** (TM {_cross:.2f} to each other; samples within a mode "
                 f"agree to at least {_spread:.2f}).")
    _sel = (f"Picking the plain sample that best fits the envelope instead of Boltz's top-confidence one gives "
            f"TM {b['tm_top_cc']:.2f} (vs {b['tm_top_conf']:.2f}).")
    mo.callout(mo.md(f"""
    **What this map shows**

    - {_acc}
    - {_conf}
    - Envelope fit of the top model: {b['cc_top_conf']:.2f} plain → {g['cc_top_conf']:.2f} guided (deposited model: {m['deposited_env_cc']:.2f}).
    - {_sel}

    Caveats: a handful of maps with {b['n']} samples per mode is a pilot, not proof. Deposited model from {m['year']};
    structures from before Boltz-2's training cutoff may be partly memorised, which inflates plain-Boltz scores.
    """), kind="success" if _d > 0.02 else "neutral")
    return


@app.cell
def _(fig_structures, mo, pick):
    mo.vstack([mo.md("## Structures: plain vs envelope-guided"), fig_structures(pick.value)])
    return


@app.cell
def _(fig_envelope, mo, pick):
    mo.vstack([mo.md("## Fit to the envelope"), fig_envelope(pick.value)])
    return


@app.cell
def _(fig_samples, mo, pick):
    mo.vstack([mo.md("## Every sample"), fig_samples(pick.value)])
    return


@app.cell
def _(C_GUIDED, RES, json, mo, np, os, pick, plt):
    _p = os.path.join(RES, "analysis", pick.value, "guide_log.json")
    if os.path.exists(_p):
        _log = json.load(open(_p))["log"]
        _fig, _a = plt.subplots(figsize=(6, 3.4))
        _a.plot([e["step"] for e in _log], [np.mean(e["cc_after"]) for e in _log], color=C_GUIDED, lw=2)
        _a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        _a.set_xlabel("Diffusion step"); _a.set_ylabel("Envelope CC (mean over samples)")
        _a.set_title(f"{pick.value}: envelope fit during guided sampling", loc="left", fontsize=10)
        _fig.tight_layout()
        _out = mo.vstack([mo.md("## Guidance at work"), _fig])
    else:
        _out = mo.md("_No guidance trace for this map._")
    _out
    return


@app.cell
def _(mo):
    show = mo.ui.multiselect(options=["deposited", "plain", "envelope-guided"],
                             value=["deposited", "plain", "envelope-guided"], label="Show")
    mo.vstack([mo.md("## 3D: top models superposed on the deposited structure"), show])
    return (show,)


@app.cell
def _(BASE, C_DEP, C_GUIDED, C_PLAIN, GUIDED, RES, mo, os, pick, show):
    try:
        import py3Dmol
    except ImportError:
        py3Dmol = None
    if py3Dmol is None:
        _out = mo.md("Install `py3Dmol` for the 3D view (`pip install py3Dmol`). The superposed files are in "
                     f"`modal_results/analysis/{pick.value}/` for ChimeraX or PyMOL.")
    else:
        _view = py3Dmol.view(width=880, height=540)
        _files = {"deposited": ("deposited.cif", C_DEP), "plain": (f"{BASE}_top.cif", C_PLAIN),
                  "envelope-guided": (f"{GUIDED}_top.cif", C_GUIDED)}
        for _i, _name in enumerate(n for n in _files if n in show.value):
            _f, _col = _files[_name]
            _view.addModel(open(os.path.join(RES, "analysis", pick.value, _f)).read(), "cif")
            _view.setStyle({"model": _i}, {"cartoon": {"color": _col, "opacity": 0.55 if _name == "deposited" else 1.0}})
        _view.zoomTo()
        # an iframe runs the page's own <script> tags (3Dmol.js); mo.Html does not
        _out = mo.vstack([mo.iframe(_view._make_html(), height="560px"),
                          mo.md("Grey = deposited, blue = plain, orange = envelope-guided. Drag to rotate.")])
    _out
    return


@app.cell
def _(FIGS, fig_adaptive, fig_envelope, fig_overview, fig_samples, fig_structures, maps, mo, os, plt):
    # Save every figure (all maps) as PNG for slides and papers.
    os.makedirs(FIGS, exist_ok=True)
    _saved = []
    _adapt = fig_adaptive()
    for _name, _fig in [("overview", fig_overview())] + ([("confidence_aware_vs_plain", _adapt)] if _adapt else []) + \
            [(f"{_e}_{_k}", _f(_e)) for _e in sorted(maps)
             for _k, _f in [("structures", fig_structures), ("envelope", fig_envelope), ("samples", fig_samples)]]:
        _path = os.path.join(FIGS, f"{_name}.png")
        _fig.savefig(_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(_fig)
        _saved.append(f"`{os.path.basename(_path)}`")
    mo.md(f"**Saved {len(_saved)} figures to** `{FIGS}`: " + ", ".join(_saved))
    return


if __name__ == "__main__":
    app.run()
