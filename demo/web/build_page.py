"""Embed demo_data.json into template.html.

Outputs
  docs/index.html                  the standalone page, served by GitHub Pages (any static host works)
  demo/web/cipherflow_demo.html    artifact body (the viewer adds the document skeleton; not tracked)

Usage (from the repository root):
    python -m demo.web.build_page
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parents[1] / "docs"

SKELETON_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Interactive demo of CipherFlow: deployment-aware planning of hybrid HE-MPC secure inference (NeurIPS 2026).">
<style>
:root { color-scheme: light; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
body { margin: 0; }
img { max-width: 100%; }
[hidden] { display: none !important; }
</style>
"""


def main() -> None:
    data = (HERE / "demo_data.json").read_text().replace("</", "<\\/")
    body = (HERE / "template.html").read_text().replace("__DEMO_DATA__", data)

    art = HERE / "cipherflow_demo.html"
    art.write_text(body)

    # Standalone: move <title>, font links and <style> into <head>; the rest goes into <body>.
    split = body.index("<div class=\"wrap\">")
    head_part, body_part = body[:split], body[split:]
    (SITE / "index.html").write_text(SKELETON_HEAD + head_part + "</head>\n<body>\n" + body_part + "\n</body>\n</html>\n")
    (SITE / ".nojekyll").write_text("")
    print(f"wrote {SITE / 'index.html'} and {art} ({len(body) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
