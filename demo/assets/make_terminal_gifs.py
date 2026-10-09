"""Render the demo's terminal sessions as animated GIFs, from real runs of the commands.

    python demo/assets/make_terminal_gifs.py          # from the repository root; needs Pillow

writes demo/assets/demo_network_given.gif (one terminal: the network given with --network) and
demo/assets/demo_network_measured.gif (two terminals: `cipherflow netbench --serve`, then the
client plans with the network it measures to it; both on this host, so the link is loopback).
The text in the frames is the commands' actual output (stderr, i.e. warnings, left out); only the
pacing -- typing, and output appearing line by line -- is staged.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 14)
CW, LH, PAD, BAR = 8.43, 18, 14, 30                 # char width, line height, padding, title bar
COLS = 112
C = {"bg": (30, 33, 39), "bar": (44, 49, 58), "fg": (215, 218, 224), "dim": (127, 132, 142),
     "user": (152, 195, 121), "path": (97, 175, 239), "edge": (60, 66, 77), "page": (22, 24, 29),
     "dot1": (224, 108, 117), "dot2": (229, 192, 123), "dot3": (152, 195, 121)}
PALETTE = Image.new("P", (1, 1))
PALETTE.putpalette([v for col in C.values() for v in col] + [0, 0, 0] * (256 - len(C)))


def run(args, timeout=600):
    env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"))
    r = subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"{args} failed:\n{r.stderr[-2000:]}")
    return r.stdout


def wrap(text):
    """Lines as a COLS-wide terminal shows them (long lines continue on the next row)."""
    rows = []
    for line in text.rstrip("\n").split("\n"):
        rows += [line[i:i + COLS] for i in range(0, len(line), COLS)] or [""]
    return rows


class Pane:
    """A terminal window: a title and `rows` lines of text, scrolled like a terminal."""

    def __init__(self, title, rows, prompt):
        self.title, self.rows, self.prompt, self.lines, self.cursor = title, rows, prompt, [], True

    def height(self):
        return BAR + PAD + self.rows * LH + PAD // 2

    def draw(self, img, top):
        d = ImageDraw.Draw(img)
        w = img.width - 2 * PAD
        d.rounded_rectangle([PAD, top, PAD + w, top + self.height()], 8, fill=C["bg"], outline=C["edge"])
        d.rounded_rectangle([PAD, top, PAD + w, top + BAR], 8, fill=C["bar"])
        d.rectangle([PAD, top + BAR - 8, PAD + w, top + BAR], fill=C["bar"])
        for i, k in enumerate(("dot1", "dot2", "dot3")):
            d.ellipse([PAD + 14 + 18 * i, top + 10, PAD + 24 + 18 * i, top + 20], fill=C[k])
        d.text((PAD + w / 2, top + BAR / 2), self.title, font=FONT, fill=C["dim"], anchor="mm")
        y0 = top + BAR + PAD // 2
        shown = self.lines[-self.rows:]
        for i, (kind, text) in enumerate(shown):
            x, y = PAD + 16, y0 + i * LH
            if kind == "cmd":
                user, path = self.prompt
                d.text((x, y), user, font=BOLD, fill=C["user"])
                x += CW * len(user)
                d.text((x, y), ":", font=FONT, fill=C["fg"])
                x += CW
                d.text((x, y), path, font=BOLD, fill=C["path"])
                x += CW * len(path)
                d.text((x, y), "$ " + text, font=FONT, fill=C["fg"])
                x += CW * (2 + len(text))
            else:
                d.text((x, y), text, font=FONT, fill=C["fg"])
                x += CW * len(text)
            if self.cursor and i == len(shown) - 1:
                d.rectangle([x + 1, y + 2, x + CW, y + LH - 2], fill=C["fg"])


class Recorder:
    def __init__(self, panes):
        self.panes, self.frames, self.durations = panes, [], []
        self.width = int(2 * PAD + 32 + COLS * CW)
        self.height = PAD + sum(p.height() + PAD for p in panes)

    def snap(self, ms):
        img = Image.new("RGB", (self.width, self.height), C["page"])
        top = PAD
        for p in self.panes:
            p.draw(img, top)
            top += p.height() + PAD
        self.frames.append(img.quantize(palette=PALETTE, dither=Image.Dither.NONE))
        self.durations.append(ms)

    def type(self, pane, cmd):
        pane.lines.append(("cmd", ""))
        self.snap(600)
        for k in range(1, len(cmd) + 1):
            pane.lines[-1] = ("cmd", cmd[:k])
            self.snap(35 if cmd[k - 1] != " " else 70)
        self.snap(450)

    def output(self, pane, rows, per_line=45, pause_after=None, pause=0):
        pane.cursor = False
        for i, r in enumerate(rows):
            pane.lines.append(("out", r))
            self.snap(pause if pause_after is not None and i == pause_after else per_line)
        pane.cursor = True

    def save(self, path):
        self.frames[0].save(path, save_all=True, append_images=self.frames[1:], duration=self.durations,
                            loop=0, optimize=False, disposal=1)
        print(f"wrote {path} ({len(self.frames)} frames, {sum(self.durations) / 1e3:.0f} s, "
              f"{path.stat().st_size / 1e6:.2f} MB)")


def given():
    cmd = "python -m demo.plan_pytorch_model --model bert --blocks 1 --network 10Mbps/1ms"
    out = wrap(run(cmd.split()[1:]))
    pane = Pane("terminal — CipherFlow demo: the network given", 34, ("user", "~/CipherFlow"))
    rec = Recorder([pane])
    rec.type(pane, cmd)
    rec.snap(1400)                                   # planning (takes about 10 s)
    rec.output(pane, out, pause_after=2, pause=3200)  # pause on the summary, then the rest scrolls in
    pane.lines.append(("cmd", ""))                   # the shell's next prompt
    rec.snap(9000)
    rec.save(OUT / "demo_network_given.gif")


def measured():
    server_cmd = "cipherflow netbench --serve --host 127.0.0.1"
    client_cmd = "python -m demo.plan_pytorch_model --model bert --blocks 1 --ip 127.0.0.1"
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    srv = subprocess.Popen([sys.executable, "-m", "cipherflow", *server_cmd.split()[1:]], cwd=REPO, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        banner = srv.stdout.readline()
        time.sleep(0.5)
        out = wrap(run(client_cmd.split()[1:]))
    finally:
        srv.terminate()
        srv.wait(timeout=10)
    server = Pane("terminal 1 — server: cipherflow netbench --serve", 4, ("server", "~/CipherFlow"))
    client = Pane("terminal 2 — client: plans with the network measured to the server", 30,
                  ("client", "~/CipherFlow"))
    rec = Recorder([server, client])
    server.cursor = False
    client.cursor = False
    rec.snap(300)
    server.cursor = True
    rec.type(server, server_cmd)
    rec.output(server, wrap(banner), per_line=200)
    server.cursor = False
    client.cursor = True
    rec.snap(900)
    rec.type(client, client_cmd)
    rec.snap(900)
    rec.output(client, out[:2], per_line=1600)        # the measurement (RTT, then 2 s each way)
    rec.output(client, out[2:], pause_after=3, pause=3200)
    client.lines.append(("cmd", ""))
    rec.snap(9000)
    rec.save(OUT / "demo_network_measured.gif")


if __name__ == "__main__":
    given()
    measured()
