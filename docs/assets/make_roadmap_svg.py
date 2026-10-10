"""Generate the roadmap figure: discriminative Transformers (done) -> generative models (next) -> a private
inference runtime (then), with the details of the next stage below: its two steps, and a LLaMA-style
decoder block marked by which operators exist today.

Writes docs/assets/cipherflow-roadmap-{light,dark}.svg in the style of the README figure.

    python docs/assets/make_roadmap_svg.py
"""
from pathlib import Path

from make_ecosystem_svg import FONT, THEMES

W, H = 1120, 600
CW, GAP, X0, CY, CH = 330, 25, 40, 104, 176      # stage cards: width, gap, left, top, height


def svg(c: dict) -> str:
    def text(x, y, s, size=15, color=c["ink"], weight=400, anchor="start"):
        return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
                f'fill="{color}" text-anchor="{anchor}">{s}</text>')

    def muted(x, y, s, size=13.5, anchor="start"):
        return text(x, y, s, size, c["muted"], anchor=anchor)

    def box(x, y, w, h, fill, stroke, sw=1.4, dash="", rx=10):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'

    def line(x1, y1, x2, y2, color, sw=2, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>'

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="CipherFlow roadmap: discriminative Transformers (done), generative models (next), a private '
         f'inference runtime (then); below, the two steps of the next stage">',
         f'<defs><marker id="ra" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c["muted"]}"/></marker></defs>']

    # ---- timeline: done, next, then
    cx = [X0 + i * (CW + GAP) + CW // 2 for i in range(3)]
    ty = 62
    p += [line(cx[0], ty, cx[1], ty, c["he"], 3), line(cx[1], ty, cx[2], ty, c["muted"], 2, "6 6"),
          f'<circle cx="{cx[0]}" cy="{ty}" r="15" fill="{c["he"]}"/>',
          f'<path d="M{cx[0] - 7},{ty} l5,5 l9,-10" fill="none" stroke="{c["box"]}" stroke-width="3" '
          f'stroke-linecap="round" stroke-linejoin="round"/>',
          f'<circle cx="{cx[1]}" cy="{ty}" r="15" fill="{c["accent_fill"]}" stroke="{c["accent"]}" stroke-width="3"/>',
          f'<circle cx="{cx[1]}" cy="{ty}" r="5.5" fill="{c["accent"]}"/>',
          f'<circle cx="{cx[2]}" cy="{ty}" r="15" fill="{c["box"]}" stroke="{c["muted"]}" stroke-width="2" stroke-dasharray="4 4"/>']
    for x, (lab, col) in zip(cx, [("DONE · v0.1–v0.4", c["he"]), ("NEXT", c["accent"]), ("THEN", c["muted"])]):
        p += [text(x, 30, lab, 14, col, 700, "middle")]

    # ---- the three stages, one card each
    stages = [
        dict(fill=c["he_fill"], stroke=c["he"], sw=1.8, dash="", n="①", title="Discriminative Transformers",
             sub="BERT-base, ViT-B/16", items=["PyTorch front end", "HE and MPC backends, measured", "plans, calibration, emulation"]),
        dict(fill=c["accent_fill"], stroke=c["accent"], sw=2.4, dash="", n="②", title="Generative models",
             sub="GPT-2, then LLaMA-style decoders", items=["backend operators for decoders", "torch.export front end", "prefill and decode plans"]),
        dict(fill=c["box"], stroke=c["muted"], sw=1.6, dash="7 6", n="③", title="Private inference runtime",
             sub="plans executed under encryption", items=["two-party runtime: keys, transport", "encrypted execution, BERT to LLMs", "GPU backends and SPU"]),
    ]
    for i, st in enumerate(stages):
        x = X0 + i * (CW + GAP)
        p += [box(x, CY, CW, CH, st["fill"], st["stroke"], st["sw"], st["dash"], 12),
              text(x + 18, CY + 34, f'{st["n"]} {st["title"]}', 19, weight=700),
              muted(x + 18, CY + 58, st["sub"], 14)]
        for k, item in enumerate(st["items"]):
            y = CY + 96 + k * 30
            p += [f'<circle cx="{x + 24}" cy="{y - 5}" r="3" fill="{st["stroke"]}"/>', text(x + 36, y, item, 15)]
        if i < 2:
            p += [f'<line x1="{x + CW + 3}" y1="{CY + CH / 2}" x2="{x + CW + GAP - 3}" y2="{CY + CH / 2}" '
                  f'stroke="{c["muted"]}" stroke-width="1.8" marker-end="url(#ra)"/>']

    # ---- details of the next stage: its two steps
    dy = CY + CH + 46
    p += [line(cx[1], CY + CH + 2, cx[1], dy - 30, c["accent"], 1.6, "3 4"),
          text(X0, dy, "Details · ② Generative models", 17, weight=700),
          muted(X0 + 286, dy, "two steps: the operators decoders need, then the models")]
    sw2, sg = (W - 2 * X0 - 34) / 2, 34
    steps = [("Step 1 · backend operators", c["accent"], "",
              ["causal mask, RoPE, RMSNorm, SwiGLU, KV cache, embedding,", "LM head, sampling: HE kernels, MPC protocols, measured costs"]),
             ("Step 2 · the models", c["muted"], "6 5",
              ["torch.export front end; prefill and decode planned separately,", "KV cache as state; per-token latency; GPT-2, then LLaMA-style"])]
    for k, (title, stroke, dash, lines_) in enumerate(steps):
        x = X0 + k * (sw2 + sg)
        p += [box(x, dy + 14, sw2, 86, c["box"], stroke, 1.6, dash, 10), text(x + 16, dy + 40, title, 15, weight=700)]
        for j, ln in enumerate(lines_):
            p += [muted(x + 16, dy + 64 + j * 20, ln, 13.5)]
        if k == 0:
            p += [f'<line x1="{x + sw2 + 4}" y1="{dy + 57}" x2="{x + sw2 + sg - 4}" y2="{dy + 57}" '
                  f'stroke="{c["muted"]}" stroke-width="1.8" marker-end="url(#ra)"/>']

    # ---- a LLaMA-style decoder block, marked by what exists today
    by = CY + CH + 46 + 136
    p += [text(X0, by, "Step 1 in one LLaMA-style decoder block", 15.5, weight=700),
          f'<rect x="{X0 + 540}" y="{by - 13}" width="16" height="16" rx="4" fill="{c["box"]}" stroke="{c["he"]}" stroke-width="1.8"/>',
          muted(X0 + 564, by, "supported today"),
          f'<rect x="{X0 + 690}" y="{by - 13}" width="16" height="16" rx="4" fill="{c["accent_fill"]}" stroke="{c["accent"]}" '
          f'stroke-width="1.8" stroke-dasharray="4 3"/>',
          muted(X0 + 714, by, "added in step 1"),
          muted(X0 + 858, by, "(execution order)")]
    rows = [
        [("Embedding", 0), ("RMSNorm", 0), ("Q/K/V proj", 1), ("RoPE", 0), ("QKᵀ", 1), ("causal mask", 0),
         ("Softmax", 1), ("Attn·V", 1), ("KV cache", 0), ("Out proj", 1), ("Add", 1)],
        [("RMSNorm", 0), ("Gate/Up proj", 1), ("SiLU", 0), ("Gate × Up", 0), ("Down proj", 1), ("Add", 1),
         ("LM head", 0), ("Sampling", 0)],
    ]
    for r, row in enumerate(rows):
        y = by + 22 + r * 62
        widths = [max(56, 20 + (7.8 if ok else 8.4) * len(name)) for name, ok in row]
        gap = (W - 2 * X0 - sum(widths)) / (len(row) - 1)
        x = X0
        for k, ((name, ok), w) in enumerate(zip(row, widths)):
            if ok:
                p += [box(x, y, w, 38, c["box"], c["he"], 1.8, rx=8),
                      text(x + w / 2, y + 24, name, 13.5, anchor="middle")]
            else:
                p += [box(x, y, w, 38, c["accent_fill"], c["accent"], 1.8, "4 3", rx=8),
                      text(x + w / 2, y + 24, name, 13.5, weight=600, anchor="middle")]
            if k < len(row) - 1:
                p += [f'<line x1="{x + w + 3}" y1="{y + 19}" x2="{x + w + gap - 3}" y2="{y + 19}" '
                      f'stroke="{c["muted"]}" stroke-width="1.4" marker-end="url(#ra)"/>']
            x += w + gap
    p.append("</svg>")
    return "\n".join(p)


def main() -> None:
    out = Path(__file__).resolve().parent
    for name, colors in THEMES.items():
        (out / f"cipherflow-roadmap-{name}.svg").write_text(svg(colors))
        print("wrote", out / f"cipherflow-roadmap-{name}.svg")


if __name__ == "__main__":
    main()
