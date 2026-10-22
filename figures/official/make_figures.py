"""Write-up figures 2-4: each source image with its caption rendered underneath.

  conda activate nsfold && python writeup_figures/make_figures.py          # all figures
  python writeup_figures/make_figures.py 4                                  # just Figure 4

Edit a caption in FIGURES below and rerun. Source images are in source/ (from the
skllpr/envelope-guided-boltz repo, figures/ at commit f2a43fd); the method diagram is drawn here.
Figure 1 (figure1.png) is the composed a/b/c figure and has no caption by design.
"""
import os, sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source")
if not os.path.isdir(SRC):
    SRC = os.path.dirname(HERE)  # repo layout: figures/official/ -> figures/
WIDTH = 2400            # final figure width in px (~6.5 in at 370 dpi)
FONT_PX = 46            # caption font size in px (~9 pt at that width)
MARGIN = 60
FONTS = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")

# number -> (source image, bold lead-in, caption body)
FIGURES = {
    2: ("method.png",
        "Envelope-guided structure prediction.",
        "Boltz-2 predicts a complex from sequence alone by iterative denoising (diffusion). At every "
        "denoising step, the current prediction is fitted into the ~20 Å negative-stain envelope and each chain "
        "is nudged as a rigid body, with a small per-residue term, to fit the envelope better; a nudge is kept "
        "only if the fit improves. Boltz-2's network is not retrained. In confidence-aware guidance, plain "
        "Boltz-2 runs first and its interface confidence (ipTM) sets the guidance strength: full below 0.5, "
        "off above 0.8. Each prediction is scored by TM-score against the atomic model the authors deposited "
        "with the map (0 to 1; above 0.5 indicates the correct overall arrangement)."),
    3: ("confidence_aware_vs_plain.png",
        "Confidence-aware guidance applies the envelope only where Boltz-2 is unsure.",
        "TM-score of plain Boltz-2 (blue circles) and confidence-aware guidance (green squares) for each map. "
        "Under each change is plain Boltz-2's ipTM and the guidance strength it sets (full below 0.5, none "
        "above 0.8, linear in between). The two large gains come from maps with low or borderline confidence "
        "(EMD-1149, ipTM 0.19; EMD-2712, ipTM 0.51), confident maps are left unchanged, and the largest loss is "
        "0.07 (EMD-5755). Mean change +0.04 TM-score."),
    4: ("EMD-1149_envelope.png",
        "Guidance moves the prediction into the measured envelope.",
        "EMD-1149 (PDB 2BYU): residues of plain Boltz-2 (blue, top) and envelope-guided Boltz-2 (orange, "
        "bottom) placed in the negative-stain envelope (light grey), shown in three orthogonal projections, "
        "with the deposited model in grey dots. Only 69% of the plain model lies inside the envelope, compared "
        "with 97% for the guided model and for the deposited model. Across all 13 maps guidance raised the "
        "mean envelope fit from 0.61 to 0.79, but accuracy improved mainly where Boltz-2 had assembled the "
        "complex wrongly: fitting the envelope is not the same as being right."),
}


def draw_method(path):
    """Flow diagram for Figure 2 (matplotlib, so text and colours are easy to change)."""
    fig, ax = plt.subplots(figsize=(13, 4.2), dpi=200)
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")
    boxes = [
        (0.2, "Inputs", "Protein sequences\n+ negative-stain map\n(~20 Å envelope)", "#EEF0EE", "#172026"),
        (3.4, "Boltz-2 diffusion", "Denoise a noisy\nstructure step by step\n(network unchanged)", "#E3ECF8", "#2A78D6"),
        (6.6, "Envelope guidance", "Fit into envelope, nudge\nchains to fit better;\nkeep only if fit improves", "#FBE6DA", "#C4521F"),
        (9.8, "Score", "TM-score vs the\ndeposited model\n(above 0.5 = correct\narrangement)", "#EEF0EE", "#172026"),
    ]
    for x, head, body, fill, edge in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.9), 2.9, 2.6, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=fill, ec=edge, lw=2))
        ax.text(x + 1.45, 3.05, head, ha="center", va="center", fontsize=13.5, fontweight="bold", color=edge)
        ax.text(x + 1.45, 1.9, body, ha="center", va="center", fontsize=12, color="#172026", linespacing=1.4)
    for x in (3.1, 9.5):
        ax.annotate("", xy=(x + 0.3, 2.2), xytext=(x, 2.2), arrowprops=dict(arrowstyle="-|>", lw=2, color="#6A747B"))
    # guidance loops back into every diffusion step
    ax.annotate("", xy=(4.85, 0.85), xytext=(8.05, 0.85),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#C4521F", connectionstyle="arc3,rad=-0.35"))
    ax.annotate("", xy=(6.6, 2.2), xytext=(6.3, 2.2), arrowprops=dict(arrowstyle="-|>", lw=2, color="#6A747B"))
    ax.text(6.45, 0.12, "every step   ·   strength set by plain ipTM (confidence-aware)", ha="center",
            fontsize=12, color="#C4521F")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def wrap(draw, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur); cur = w
    return lines + [cur]


def build(n):
    src, lead, body = FIGURES[n]
    path = os.path.join(SRC, src)
    if src == "method.png":
        path = os.path.join(HERE, "method.png")
        draw_method(path)
    im = Image.open(path).convert("RGB")
    im = im.resize((WIDTH - 2 * MARGIN, round(im.height * (WIDTH - 2 * MARGIN) / im.width)), Image.LANCZOS)
    bold = ImageFont.truetype(os.path.join(FONTS, "DejaVuSans-Bold.ttf"), FONT_PX)
    reg = ImageFont.truetype(os.path.join(FONTS, "DejaVuSans.ttf"), FONT_PX)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    head = f"Figure {n}. {lead}"
    text_w = WIDTH - 2 * MARGIN
    # bold lead-in followed by regular text on the same lines
    words = [(w, bold) for w in head.split()] + [(w, reg) for w in body.split()]
    lines, cur, cur_w = [], [], 0
    space = probe.textlength(" ", font=reg)
    for w, f in words:
        wl = probe.textlength(w, font=f)
        if cur and cur_w + space + wl > text_w:
            lines.append(cur); cur, cur_w = [], 0
        cur.append((w, f)); cur_w += (space if cur_w else 0) + wl
    lines.append(cur)
    line_h = round(FONT_PX * 1.4)
    cap_h = len(lines) * line_h
    out = Image.new("RGB", (WIDTH, MARGIN + im.height + MARGIN + cap_h + MARGIN), "white")
    out.paste(im, (MARGIN, MARGIN))
    d = ImageDraw.Draw(out)
    y = MARGIN + im.height + MARGIN
    for line in lines:
        x = MARGIN
        for w, f in line:
            d.text((x, y), w, font=f, fill=(23, 32, 38))
            x += probe.textlength(w, font=f) + space
        y += line_h
    dst = os.path.join(HERE, f"figure{n}.png")
    out.save(dst, dpi=(370, 370))
    print("wrote", dst, out.size)


if __name__ == "__main__":
    for n in [int(a) for a in sys.argv[1:]] or sorted(FIGURES):
        build(n)
