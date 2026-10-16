"""Generate the README figure: where CipherFlow sits in the private-inference stack.

Writes docs/assets/cipherflow-ecosystem.svg, which follows the viewer's light or dark color scheme
(its colors are CSS variables switched by prefers-color-scheme inside the file, on its own background),
and cipherflow-ecosystem-light.svg, always light, for the white website. Comic Sans MS with fallbacks.

    python docs/assets/make_ecosystem_svg.py
"""
import re
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

W, H = 1120, 990
X, BW = 290, 580          # main column
CX = X + BW // 2
HEAD = 44                 # header band height


def svg(c: dict) -> str:
    def text(x, y, s, size=16, color=c["ink"], weight=400, anchor="start"):
        return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
                f'fill="{color}" text-anchor="{anchor}">{s}</text>')

    def muted(x, y, s, size=14, anchor="start"):
        return text(x, y, s, size, c["muted"], anchor=anchor)

    def box(x, y, w, h, fill, stroke, sw=1.4, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'

    def layer(x, y, w, h, title, fill, stroke, sw=1.4):
        """Block split by a divider: layer name on top, contents below."""
        return [box(x, y, w, h, fill, stroke, sw),
                f'<line x1="{x}" y1="{y + HEAD}" x2="{x + w}" y2="{y + HEAD}" stroke="{stroke}" stroke-width="{sw}"/>',
                text(x + w / 2, y + 30, title, 21, weight=700, anchor="middle")]

    def arrow(x1, y1, x2, y2):
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c["ink"]}" stroke-width="1.8" '
                f'marker-end="url(#ah)"/>')

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'role="img" aria-label="CipherFlow in the private-inference stack: inputs and outputs">',
         f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c["ink"]}"/></marker></defs>']

    # Model layer: a PyTorch model, traced into the operator graph
    y = 20
    p += layer(X, y, BW, 128, "Model Layer", c["box"], c["line"])
    p += [text(CX, y + HEAD + 28, "PyTorch model → operator graph", 18, weight=600, anchor="middle"),
          muted(CX, y + HEAD + 52, "nn.Module, e.g. demo/transformer.py (BERT-base, ViT-B/16)", anchor="middle"),
          muted(CX, y + HEAD + 74, "traced by cf.models.from_torch into operators + residuals", anchor="middle")]

    # (1) graph
    p += [arrow(CX, 148, CX, 204), text(CX + 14, 172, "① graph G = (V, E)", 18, weight=600),
          muted(CX + 14, 194, "operators · tensor shapes · residual edges")]

    # Planning layer
    y = 206
    p += layer(X, y, BW, 262, "Planning Layer", c["accent_fill"], c["accent"], sw=2.2)
    p += [text(CX, y + HEAD + 36, "CipherFlow", 24, weight=700, anchor="middle")]
    cw, gap, cy = 168, 26, y + HEAD + 54
    x0 = CX - (3 * cw + 2 * gap) // 2
    for i, (name, what) in enumerate([("MicroBench", "measures missing costs"),
                                      ("State expansion", "domain × CKKS level"),
                                      ("BlockFold", "folds residual blocks")]):
        x = x0 + i * (cw + gap)
        p += [box(x, cy, cw, 64, c["box"], c["line"]),
              text(x + cw / 2, cy + 27, name, 16, weight=600, anchor="middle"),
              muted(x + cw / 2, cy + 49, what, 13, anchor="middle")]
        if i:
            p += [arrow(x - gap + 3, cy + 32, x - 3, cy + 32)]
    p += [text(CX, y + 262 - 44, "plan = cf.compile(graph, deployment, profile, calibration=None)", 15,
               weight=600, anchor="middle"),
          muted(CX, y + 262 - 18, "returns plan.json · plan.explain(steps=True) · plan.steps()", anchor="middle")]

    # (2) deployment (side input)
    dx, dy, dw = 14, 238, 256
    p += layer(dx, dy, dw, 196, "Deployment", c["box"], c["line"])
    p += [text(dx + 14, dy + HEAD + 28, "② D = (HW, BW, RTT)", 17, weight=600),
          muted(dx + 14, dy + HEAD + 52, "HW: cpu | gpu profile,"),
          muted(dx + 30, dy + HEAD + 72, "or this machine (MicroBench)"),
          muted(dx + 14, dy + HEAD + 96, "BW/RTT: --network 10Mbps/1ms,"),
          muted(dx + 30, dy + HEAD + 116, "or measured: --ip SERVER"),
          muted(dx + 14, dy + HEAD + 140, "security: 2pc | 2pc-dealer"),
          arrow(dx + dw, dy + 98, X - 2, dy + 98)]

    # (4) plan down, (3) methods + costs up
    ax4, ax3 = X + 110, X + BW - 150
    p += [arrow(ax4, 468, ax4, 576), text(ax4 + 14, 500, "④ plan.json", 18, weight=600),
          muted(ax4 + 14, 522, "each operator: HE | MPC, level, kernel"),
          muted(ax4 + 14, 542, "each edge: conversions, bootstraps"),
          arrow(ax3, 576, ax3, 470), text(ax3 + 14, 500, "③ methods + measured costs", 18, weight=600),
          muted(ax3 + 14, 522, "per operator and domain: time,"),
          muted(ax3 + 14, 542, "bytes, rounds, CKKS levels used")]

    # Backend layer
    y = 578
    p += layer(X, y, BW, 196, "Backend Layer", c["box"], c["line"])
    sw_, sg = 272, 16
    sx = CX - sw_ - sg // 2
    for i, (fill, stroke, t1, t2, t3) in enumerate([
            (c["he_fill"], c["he"], "HE: OpenFHE · NEXUS", "OpenFHE CKKS (CPU)", "NEXUS: SEAL (CPU), Phantom (CUDA)"),
            (c["mpc_fill"], c["mpc"], "MPC: SCI · SHAFT", "SCI/BOLT: 2PC (CPU)", "SHAFT: 2PC + dealer (CPU, GPU)")]):
        x = sx + i * (sw_ + sg)
        p += [box(x, y + HEAD + 14, sw_, 96, fill, stroke, sw=1.8),
              text(x + sw_ / 2, y + HEAD + 44, t1, 17, weight=600, anchor="middle"),
              muted(x + sw_ / 2, y + HEAD + 68, t2, anchor="middle"),
              muted(x + sw_ / 2, y + HEAD + 90, t3, anchor="middle")]
    p += [muted(CX, y + 196 - 16, "HE ⇄ MPC conversions (hardened for OpenFHE) · bootstrapping", anchor="middle")]

    # Execution / serving
    p += [arrow(CX, 774, CX, 808), muted(CX + 14, 796, "runs the plan")]
    y = 810
    p += layer(X, y, BW, 160, "Execution / Serving", "none", c["line"])
    p += [text(CX, y + HEAD + 28, "cipherflow.runtime: the plan on real inputs", 16, weight=600, anchor="middle"),
          muted(CX, y + HEAD + 52, "check_plan (domains, levels) · calibrate (value ranges)", anchor="middle"),
          muted(CX, y + HEAD + 74, "emulate: error vs PyTorch, GLUE accuracy (no cryptography)", anchor="middle"),
          muted(CX, y + HEAD + 98, "encrypted two-party execution: planned (ROADMAP.md §6)", anchor="middle")]

    p.append("</svg>")
    return "\n".join(p)


BACKGROUND = {"light": "#ffffff", "dark": "#0d1117"}      # the adaptive file's own ground (GitHub's canvases)


def _css(theme: str) -> str:
    return "; ".join(f"--{k.replace('_', '-')}: {v}" for k, v in {**THEMES[theme], "bg": BACKGROUND[theme]}.items())


def adaptive(draw) -> str:
    """The figure `draw(colors)` with every color a CSS variable, set for light and dark inside the file.
    Colors move from presentation attributes into style attributes, where var() works in every browser."""
    body = draw({k: f"var(--{k.replace('_', '-')})" for k in THEMES["light"]})

    def to_style(m):
        tag = m.group(0)
        props = re.findall(r' (fill|stroke)="(var\(--[\w-]+\))"', tag)
        if not props:
            return tag
        tag = re.sub(r' (fill|stroke)="var\(--[\w-]+\)"', "", tag)
        end = -2 if tag.endswith("/>") else -1
        return tag[:end] + ' style="' + ";".join(f"{a}:{v}" for a, v in props) + '"' + tag[end:]
    body = re.sub(r"<[a-zA-Z][^>]*>", to_style, body)
    head = (f"<style>svg {{ {_css('light')} }} "
            f"@media (prefers-color-scheme: dark) {{ svg {{ {_css('dark')} }} }}</style>"
            '<rect width="100%" height="100%" style="fill:var(--bg)"/>')
    i = body.index(">", body.index("<svg")) + 1
    return body[:i] + head + body[i:]


def write_figure(stem: str, draw) -> None:
    """docs/assets/<stem>.svg (adaptive) and <stem>-light.svg (always light, for the white website)."""
    out = Path(__file__).resolve().parent
    (out / f"{stem}.svg").write_text(adaptive(draw))
    (out / f"{stem}-light.svg").write_text(draw(THEMES["light"]))
    print("wrote", out / f"{stem}.svg", "and", out / f"{stem}-light.svg")


def main() -> None:
    write_figure("cipherflow-ecosystem", svg)


if __name__ == "__main__":
    main()
