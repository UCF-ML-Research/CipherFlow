"""Embed demo_data.json into template.html.

Outputs
  cipherflow/demo/cipherflow_demo.html   artifact body (the viewer adds the document skeleton)
  cipherflow/demo/site/index.html        standalone page for GitHub Pages / any static host

Usage (from the repository root):
    python -m cipherflow.demo.build_page
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

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
    site = HERE / "site"
    site.mkdir(exist_ok=True)
    (site / "index.html").write_text(SKELETON_HEAD + head_part + "</head>\n<body>\n" + body_part + "\n</body>\n</html>\n")
    (site / ".nojekyll").write_text("")
    print(f"wrote {art} and {site / 'index.html'} ({len(body) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
