"""Generate the README figure: where CipherFlow sits in the private-inference stack.

Writes docs/assets/cipherflow-ecosystem-{light,dark}.svg (transparent background,
Comic Sans MS with fallbacks, so the files render as plain images on GitHub).

    python docs/assets/make_ecosystem_svg.py
"""
from pathlib import Path

THEMES = {
    "light": dict(ink="#1f2328", muted="#59636e", line="#d1d9e0", box="#f6f8fa",
                  accent="#0969da", accent_fill="#ddf4ff", he="#1a8f6e", mpc="#eb6834",
                  he_fill="#dcf3ea", mpc_fill="#fde9e0"),
    "dark": dict(ink="#e6edf3", muted="#9198a1", line="#3d444d", box="#151b23",
                 accent="#4493f8", accent_fill="#0d2240", he="#2fb38c", mpc="#d95926",
                 he_fill="#0e2e25", mpc_fill="#3a1d12"),
}
FONT = "'Comic Sans MS', 'Comic Sans', 'Comic Neue', 'Chalkboard SE', cursive"

W, H = 1100, 840
X, BW = 270, 580          # main column
CX = X + BW // 2
HEAD = 44                 # header band height


def svg(c: dict) -> str:
    def text(x, y, s, size=16, color=c["ink"], weight=400, anchor="start"):
        return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
                f'fill="{color}" text-anchor="{anchor}">{s}</text>')

    def box(x, y, w, h, fill, stroke, sw=1.4, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'

    def layer(x, y, w, h, title, fill, stroke, sw=1.4, dash=""):
        """Block split by a divider: layer name on top, contents below."""
        return [box(x, y, w, h, fill, stroke, sw, dash),
                f'<line x1="{x}" y1="{y + HEAD}" x2="{x + w}" y2="{y + HEAD}" stroke="{stroke}" stroke-width="{sw}"'
                + (f' stroke-dasharray="{dash}"' if dash else "") + '/>',
                text(x + w / 2, y + 30, title, 21, weight=700, anchor="middle")]

    def arrow(x1, y1, x2, y2):
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c["ink"]}" stroke-width="1.8" '
                f'marker-end="url(#ah)"/>')

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'role="img" aria-label="CipherFlow in the private-inference stack: inputs and outputs">',
         f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c["ink"]}"/></marker></defs>']

    # Model layer
    y = 20
    p += layer(X, y, BW, 100, "Model Layer", c["box"], c["line"])
    p += [text(CX, y + HEAD + 36, "Model graph / IR", 18, weight=600, anchor="middle")]

    # (1) graph
    p += [arrow(CX, 120, CX, 176), text(CX + 14, 155, "① Graph G = (V, E)", 18, weight=600)]

    # Planning layer
    y = 178
    p += layer(X, y, BW, 236, "Planning Layer", c["accent_fill"], c["accent"], sw=2.2)
    p += [text(CX, y + HEAD + 40, "CipherFlow", 24, weight=700, anchor="middle")]
    cw, gap, cy = 164, 30, y + HEAD + 62
    x0 = CX - (3 * cw + 2 * gap) // 2
    for i, name in enumerate(["MicroBench", "State expansion", "BlockFold"]):
        x = x0 + i * (cw + gap)
        p += [box(x, cy, cw, 54, c["box"], c["line"]),
              text(x + cw / 2, cy + 33, name, 17, weight=600, anchor="middle")]
        if i:
            p += [arrow(x - gap + 3, cy + 27, x - 3, cy + 27)]
    p += [text(CX, y + 236 - 24, "plan = cipherflow.compile(graph, deployment, profile)", 17,
               weight=600, anchor="middle")]

    # (2) deployment (side input)
    dy = 224
    p += layer(20, dy, 222, 146, "Deployment", c["box"], c["line"])
    p += [text(36, dy + HEAD + 32, "② D = (HW, BW, RTT)", 17, weight=600),
          text(36, dy + HEAD + 60, "hardware: CPU or GPU", 15, c["muted"]),
          text(36, dy + HEAD + 84, "network bandwidth · RTT", 15, c["muted"]),
          arrow(242, dy + 73, X - 2, dy + 73)]

    # (4) plan down, (3) methods + costs up
    p += [arrow(X + 120, 414, X + 120, 508), text(X + 134, 468, "④ deploy plan", 18, weight=600),
          arrow(X + BW - 150, 508, X + BW - 150, 416), text(X + BW - 136, 468, "③ methods + measured costs", 18, weight=600)]

    # Backend layer
    y = 510
    p += layer(X, y, BW, 156, "Backend Layer", c["box"], c["line"])
    sw_, sg = 262, 20
    sx = CX - sw_ - sg // 2
    for i, (fill, stroke, t1, t2) in enumerate([(c["he_fill"], c["he"], "HE: SEAL · Phantom", "CPU · CUDA"),
                                                (c["mpc_fill"], c["mpc"], "MPC: SCI · CrypTen", "CPU · GPU")]):
        x = sx + i * (sw_ + sg)
        p += [box(x, y + HEAD + 16, sw_, 80, fill, stroke, sw=1.8),
              text(x + sw_ / 2, y + HEAD + 50, t1, 17, weight=600, anchor="middle"),
              text(x + sw_ / 2, y + HEAD + 76, t2, 15, c["muted"], anchor="middle")]

    # Execution / serving
    p += [arrow(CX, 666, CX, 702)]
    y = 704
    p += layer(X, y, BW, 116, "Execution / Serving", "none", c["line"], dash="6 5")
    p += [text(CX, y + HEAD + 32, "runtime consume plan", 17, anchor="middle"),
          text(CX, y + HEAD + 58, "runtime adapter", 17, anchor="middle")]

    p.append("</svg>")
    return "\n".join(p)


def main() -> None:
    out = Path(__file__).resolve().parent
    for name, colors in THEMES.items():
        (out / f"cipherflow-ecosystem-{name}.svg").write_text(svg(colors))
        print("wrote", out / f"cipherflow-ecosystem-{name}.svg")


if __name__ == "__main__":
    main()
