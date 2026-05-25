"""Render a publication-style architecture diagram of the band-gated MobileNetV2-UNet.

Top: main U-Net flow with the prior-free band gate up front (M_k = F ∘ G_k).
Bottom-left: the gate internals (hard top-k forward + straight-through backward).
Bottom-right: the U-Net decoder block and the MobileNetV2 inverted-residual block.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon

C = dict(
    gate="#E8924A", enc="#5B8FB9", dec="#54B3A0", bott="#37598A", head="#9AA0A6",
    io_face="#E76F51", io_top="#F2A488", io_side="#C75B41",
    panel_gate="#F7E6D2", panel_dec="#ECECEC", cell="#CFE3F2", cellon="#E8924A",
    act="#E76F51", line="#333333", skip="#888888", up="#2E8B57", down="#C0392B",
)


def cube(ax, x, y, s=0.34, d=0.16):
    """Small isometric cube to denote a spectral data tensor."""
    ax.add_patch(Polygon([(x, y), (x + s, y), (x + s, y + s), (x, y + s)],
                         closed=True, fc=C["io_face"], ec="#222", lw=1))
    ax.add_patch(Polygon([(x, y + s), (x + d, y + s + d), (x + s + d, y + s + d),
                          (x + s, y + s)], closed=True, fc=C["io_top"], ec="#222", lw=1))
    ax.add_patch(Polygon([(x + s, y), (x + s + d, y + d), (x + s + d, y + s + d),
                          (x + s, y + s)], closed=True, fc=C["io_side"], ec="#222", lw=1))


def box(ax, x, y, w, h, text, fc, tc="white", fs=8.5, ec="#333", lw=1.2,
        dashed=False, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.015,rounding_size=0.07",
                 fc=fc, ec=ec, lw=lw, linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal")


def arrow(ax, p1, p2, color="#333", ls="-", lw=1.5, scale=11, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=scale,
                 color=color, ls=ls, lw=lw,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def opnode(ax, x, y, sym, r=0.17, fc="white", fs=11):
    ax.add_patch(Circle((x, y), r, fc=fc, ec="#333", lw=1.3, zorder=6))
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs, zorder=7)


def cells(ax, x, y, n, on, cw=0.26, ch=0.34, gap=0.04):
    for i in range(n):
        cx = x + i * (cw + gap)
        ax.add_patch(Rectangle((cx, y), cw, ch, fc=(C["cellon"] if i in on else C["cell"]),
                     ec="#444", lw=0.8))
    return x + n * (cw + gap) - gap  # right edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/figures/architecture.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis("off")
    ax.text(8, 10.72, "Architecture of the band-gated MobileNetV2-UNet  "
            r"($\mathcal{M}_k=\mathcal{F}\circ\mathcal{G}_k$)",
            ha="center", va="center", fontsize=15, fontweight="bold")

    # ---------------- main flow (U-Net) ----------------
    xe, xd, w, h = 3.55, 10.75, 1.75, 0.62
    yL = [9.55, 8.45, 7.35, 6.25]          # levels L1..L4
    enc_lab = ["Stage 1\n16 ch · 1/2", "Stage 2\n24 ch · 1/4",
               "Stage 3\n32 ch · 1/8", "Stage 4\n96 ch · 1/16"]
    dec_lab = ["Dec → 16 ch", "Dec → 24 ch", "Dec → 32 ch", "Dec → 96 ch"]

    # input cube + gate
    cube(ax, 0.45, 9.62)
    ax.text(0.62, 9.46, r"$X$  ($B{\times}H{\times}W$)", ha="center", va="top", fontsize=8)
    box(ax, 1.45, 9.5, 1.85, 0.72, "Band Gate\n" r"$\mathcal{G}_k$ (select k of B)",
        C["gate"], dashed=True, fs=8.5)
    arrow(ax, (0.95, 9.79), (1.45, 9.86))
    arrow(ax, (3.30, 9.86), (xe, 9.86)); ax.text((3.30 + xe) / 2, 10.04, r"$k{\times}H{\times}W$",
            ha="center", fontsize=7.5)

    # encoder column
    for i in range(4):
        box(ax, xe, yL[i], w, h, enc_lab[i], C["enc"])
    # bottleneck
    xb, yb = 6.2, 5.05
    box(ax, xb, yb, w, h, "Bottleneck S5\n320 ch · 1/32", C["bott"])
    # decoder column
    for i in range(4):
        box(ax, xd, yL[i], w, h, dec_lab[i], C["dec"])
    # head + output
    box(ax, 12.75, 9.55, 1.5, h, "1×1 conv\n+ upsample", C["head"])
    cube(ax, 14.7, 9.62)
    ax.text(14.87, 9.46, r"bruise mask  ($\hat Y$)", ha="center", va="top", fontsize=8)

    # encoder downsample arrows
    for i in range(3):
        arrow(ax, (xe + w / 2, yL[i]), (xe + w / 2, yL[i + 1] + h), color=C["down"])
        ax.text(xe + w / 2 + 0.12, (yL[i] + yL[i + 1] + h) / 2, r"$\downarrow2$",
                color=C["down"], fontsize=8, va="center")
    arrow(ax, (xe + w / 2, yL[3]), (xb + w / 2, yb + h), color=C["down"], rad=-0.15)
    # decoder upsample arrows
    arrow(ax, (xb + w / 2, yb + h), (xd + w / 2, yL[3]), color=C["up"], rad=0.15)
    for i in range(3, 0, -1):
        arrow(ax, (xd + w / 2, yL[i]), (xd + w / 2, yL[i - 1]), color=C["up"])
        ax.text(xd + w / 2 + 0.12, (yL[i] + yL[i - 1]) / 2, r"$\uparrow2$",
                color=C["up"], fontsize=8, va="center")
    arrow(ax, (xd + w, 9.86), (12.75, 9.86))
    arrow(ax, (14.25, 9.86), (14.85, 9.86))

    # skip connections + concat nodes
    for i in range(4):
        yc = yL[i] + h / 2
        opnode(ax, xd - 0.42, yc, "C", r=0.16, fs=9)
        arrow(ax, (xe + w, yc), (xd - 0.58, yc), color=C["skip"], ls="--", lw=1.3)
        arrow(ax, (xd - 0.26, yc), (xd, yc), color=C["skip"], lw=1.3)
    ax.text((xe + w + xd) / 2, yL[0] + h / 2 + 0.16, "skip (concat)",
            ha="center", fontsize=7.5, color=C["skip"])

    # ---------------- legend ----------------
    lx, ly = 0.35, 6.35
    ax.add_patch(FancyBboxPatch((lx, ly), 2.55, 2.25, boxstyle="round,pad=0.04",
                 fc="white", ec="#555", lw=1.1))
    ax.text(lx + 1.27, ly + 2.05, "Legend", ha="center", fontsize=9, fontweight="bold")
    items = [(C["down"], r"$\downarrow$  downsample (stride 2)"),
             (C["up"], r"$\uparrow$  upsample"),
             (None, "C  concatenate (skip)"),
             (None, r"$\odot$  element-wise mask (gate)"),
             (C["enc"], "encoder / MBConv"),
             (C["dec"], "decoder block"),
             (C["gate"], "band gate")]
    for j, (col, txt) in enumerate(items):
        yy = ly + 1.72 - j * 0.26
        if col:
            ax.add_patch(Rectangle((lx + 0.12, yy - 0.07), 0.22, 0.15, fc=col, ec="#333", lw=0.7))
        ax.text(lx + 0.42, yy, txt, ha="left", va="center", fontsize=7.6)

    # ================= bottom-left: GATE detail =================
    gx, gy, gw, gh = 0.35, 0.35, 7.35, 3.55
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0.05",
                 fc=C["panel_gate"], ec="#B07A45", lw=1.4))
    ax.text(gx + gw / 2, gy + gh - 0.22, "Band-Selection Gate  "
            r"$\mathcal{G}_k$  (prior-free, hard top-k + straight-through)",
            ha="center", fontsize=10, fontweight="bold")

    # forward path
    yf = gy + 2.35
    ax.text(gx + 0.55, yf + 0.55, r"$\theta$ (B scores)", ha="center", fontsize=8)
    r0 = cells(ax, gx + 0.25, yf, 7, [])
    arrow(ax, (r0 + 0.05, yf + 0.17), (gx + 2.5, yf + 0.17))
    box(ax, gx + 2.5, yf - 0.02, 0.95, 0.4, "Top-k", C["enc"], fs=8)
    arrow(ax, (gx + 3.45, yf + 0.17), (gx + 3.75, yf + 0.17))
    ax.text(gx + 4.55, yf + 0.55, "hard mask m (k-hot)", ha="center", fontsize=8)
    rc = cells(ax, gx + 3.78, yf, 7, [1, 3, 5])
    opnode(ax, gx + 6.35, yf + 0.17, r"$\odot$")
    arrow(ax, (rc + 0.02, yf + 0.17), (gx + 6.18, yf + 0.17))
    box(ax, gx + 6.6, yf - 0.05, 0.78, 0.46, "gated\n" r"$X'$", C["gate"], fs=8)
    arrow(ax, (gx + 6.52, yf + 0.17), (gx + 6.6, yf + 0.17))
    ax.text(gx + 6.35, yf - 0.42, "input X", ha="center", fontsize=7)
    arrow(ax, (gx + 6.35, yf - 0.28), (gx + 6.35, yf + 0.01), color=C["skip"])

    # backward (STE) path
    yb2 = gy + 1.1
    box(ax, gx + 1.7, yb2 - 0.02, 1.25, 0.42, r"$g=\sigma(\theta/\tau)$", "#BBD3E6",
        tc="#222", fs=8)
    box(ax, gx + 3.4, yb2 - 0.02, 2.5, 0.42,
        r"$\tilde m = m + g - \mathrm{sg}(g)$", "#D9C2A6", tc="#222", fs=8)
    arrow(ax, (gx + 2.95, yb2 + 0.19), (gx + 3.4, yb2 + 0.19), color="#C0392B", ls="--")
    # gradient loop back to theta
    arrow(ax, (gx + 3.4, yb2 + 0.42), (gx + 0.7, yf - 0.05), color="#C0392B", ls="--", rad=0.3)
    ax.text(gx + 2.0, yb2 - 0.28,
            r"backward: soft gradient to ALL bands  (explore)", fontsize=7.3, color="#C0392B")
    ax.text(gx + 0.2, gy + 0.28,
            r"forward: strict k-hot (train $=$ deploy)   |   "
            r"$\tau$: 1.0$\rightarrow$0.05 cosine (explore$\rightarrow$commit)   |   "
            r"$k=B \Rightarrow$ identity (full-input baseline)",
            fontsize=7.5, ha="left")

    # ================= bottom-right: blocks =================
    bx, by, bw, bh = 8.0, 0.35, 7.65, 3.55
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.05",
                 fc=C["panel_dec"], ec="#777", lw=1.4))
    ax.text(bx + bw / 2, by + bh - 0.22, "U-Net decoder block   &   MobileNetV2 inverted residual (encoder)",
            ha="center", fontsize=10, fontweight="bold")

    # decoder block row
    yd = by + 2.35
    seq = [("Upsample ×2", C["dec"]), ("C concat\nskip", "#9AA0A6"),
           ("Conv 3×3\nBN, ReLU", C["enc"]), ("Conv 3×3\nBN, ReLU", C["enc"])]
    xx = bx + 0.35
    prev = None
    for txt, col in seq:
        box(ax, xx, yd, 1.45, 0.62, txt, col, fs=7.8)
        if prev is not None:
            arrow(ax, (prev, yd + 0.31), (xx, yd + 0.31))
        prev = xx + 1.45
        xx += 1.85
    ax.text(bx + 0.35, yd + 0.78, "decoder block", fontsize=8, fontweight="bold")

    # MBConv row
    ym = by + 0.85
    seq2 = [("1×1\nexpand", C["enc"]), ("DWConv\n3×3", C["enc"]), ("1×1\nproject", C["enc"])]
    xx = bx + 0.55
    prev = None
    for txt, col in seq2:
        box(ax, xx, ym, 1.3, 0.6, txt, col, fs=7.8)
        if prev is not None:
            arrow(ax, (prev, ym + 0.3), (xx, ym + 0.3))
        prev = xx + 1.3
        xx += 1.7
    opnode(ax, xx + 0.1, ym + 0.3, r"$\oplus$")
    arrow(ax, (prev, ym + 0.3), (xx - 0.07, ym + 0.3))
    arrow(ax, (bx + 0.55, ym + 0.6), (xx + 0.1, ym + 0.47), color=C["skip"], ls="--", rad=-0.35)
    ax.text(bx + 0.55, ym + 0.82, "MBConv (inverted residual)", fontsize=8, fontweight="bold")
    ax.text(xx + 0.45, ym + 0.3, "residual", fontsize=7, color=C["skip"], va="center")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
