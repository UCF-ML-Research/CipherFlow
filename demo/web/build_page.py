"""Build the website and the plan visualization from demo_data.json.

Outputs
  docs/index.html                  the CipherFlow website (site_template.html, with one BERT-base block's
                                   plans at RTT 40 ms taken from demo_data.json)
  docs/planner/index.html          the plan visualization (template.html with all of demo_data.json)
  demo/web/cipherflow_demo.html    the visualization as an artifact body (the viewer adds the document
                                   skeleton; not tracked)
  demo/web/cipherflow_site.html    the website as an artifact body (not tracked)

GitHub Pages serves docs/; any static host works.

Usage (from the repository root):
    python -m demo.web.build_page
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[1] / "docs"
SITE_RTT = 40.0

SKELETON_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="{description}">
<style>
:root {{ color-scheme: light; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }}
body {{ margin: 0; }}
img {{ max-width: 100%; }}
[hidden] {{ display: none !important; }}
</style>
"""


def standalone(body: str, marker: str, description: str) -> str:
    """A full HTML document: <title>, font links and <style> go into <head>; the rest into <body>."""
    split = body.index(marker)
    return (SKELETON_HEAD.format(description=description) + body[:split] + "</head>\n<body>\n" + body[split:]
            + "\n</body>\n</html>\n")


def site_data(data: dict) -> dict:
    """One BERT-base block (with the client's input) at RTT 40 ms, CPU and GPU, every bandwidth."""
    model = data["models"]["bert"]
    idx = [i for i, n in enumerate(model["nodes"]) if n["group"] in ("input", "block 1")]
    pos = {i: k for k, i in enumerate(idx)}
    plans = {}
    for hw in ("cpu", "gpu"):
        rows = []
        for bw in data["bandwidths"]:
            p = data["plans"][f"bert|{hw}|{bw}|{SITE_RTT:g}"]
            hyb = min(("BumbleBee", "BOLT", "BLB"), key=lambda k: p["base"][k])
            one = min(("All-HE", "All-MPC"), key=lambda k: p["base"][k])
            rows.append({
                "dom": "".join(p["dom"][i] for i in idx),
                "act": [[pos[a[1]], a[0], a[2], a[3]] for a in p["act"] if a[1] in pos],
                "cf": p["cost"]["total_ms"], "hyb": p["base"][hyb], "hybName": hyb,
                "one": p["base"][one], "oneName": "all-HE" if one == "All-HE" else "all-MPC",
            })
        plans[hw] = rows
    return {"version": data["cipherflow_version"], "bandwidths": data["bandwidths"], "rtt": SITE_RTT,
            "hardware": {"cpu": "CPU", "gpu": "GPU"},
            "ops": [model["nodes"][i]["label"] for i in idx], "src": [model["nodes"][i]["src"] for i in idx],
            "plans": plans}


def main() -> None:
    raw = (HERE / "demo_data.json").read_text()
    data = json.loads(raw)

    body = (HERE / "template.html").read_text().replace("__DEMO_DATA__", raw.replace("</", "<\\/"))
    (HERE / "cipherflow_demo.html").write_text(body)
    planner = DOCS / "planner"
    planner.mkdir(exist_ok=True)
    (planner / "index.html").write_text(standalone(
        body, '<div class="wrap">',
        "Interactive demo of CipherFlow: deployment-aware planning of hybrid HE-MPC secure inference (NeurIPS 2026)."))

    site = json.dumps(site_data(data), separators=(",", ":")).replace("</", "<\\/")
    page = (HERE / "site_template.html").read_text().replace("__SITE_DATA__", site)
    (HERE / "cipherflow_site.html").write_text(page)
    (DOCS / "index.html").write_text(standalone(
        page, '<header class="top"',
        "CipherFlow: a hardware-aware compiler framework for efficient secure inference. It places every operator "
        "of a PyTorch model in homomorphic encryption or secure multi-party computation for your hardware and network."))
    (DOCS / ".nojekyll").write_text("")
    print(f"wrote {DOCS / 'index.html'} ({len(page) / 1e3:.0f} kB), {planner / 'index.html'} "
          f"({len(body) / 1e6:.2f} MB) and {HERE / 'cipherflow_demo.html'}")


if __name__ == "__main__":
    main()
