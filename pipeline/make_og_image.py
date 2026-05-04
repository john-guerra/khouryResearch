"""Generate the social-card image used by Open Graph and Twitter previews.

Output: public/og-image.png (1200×630). Uses Pillow (already a dep) and the
app's color palette so the card reads like the live UI without needing a
screenshot of a possibly-unstable layout.
"""
from __future__ import annotations
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "public" / "og-image.png"

W, H = 1200, 630
BG = (14, 17, 22)              # body bg
PANEL = (24, 30, 38)
TEXT = (236, 240, 245)
MUTED = (160, 170, 184)
ACCENT_GG = (110, 168, 254)    # give→get blue
ACCENT_TOPIC = (199, 146, 234) # purple
ACCENT_KW = (143, 211, 145)    # green
NODE = (110, 168, 254)


def find_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    """Pick the first font that loads. Falls back to default bitmap font."""
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# macOS-ish first, then linux-ish, then anything Pillow can find.
TITLE_FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNSRounded.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]
BODY_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
]

title_font = find_font(TITLE_FONT_CANDIDATES, 64)
sub_font = find_font(BODY_FONT_CANDIDATES, 28)
tag_font = find_font(BODY_FONT_CANDIDATES, 20)
url_font = find_font(BODY_FONT_CANDIDATES, 22)


def draw_pill(draw: ImageDraw.ImageDraw, xy, text, font, fg, bg):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 14, 8
    draw.rounded_rectangle(
        (x, y, x + tw + 2 * pad_x, y + th + 2 * pad_y),
        radius=18, fill=bg,
    )
    draw.text((x + pad_x - bbox[0], y + pad_y - bbox[1]), text, font=font, fill=fg)


def render():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Decorative network on the right side. Same visual vocabulary as the app
    # (photo-circle nodes + thin colored edges) but without trying to fake a
    # screenshot of the actual graph.
    rng = random.Random(7)
    nodes: list[tuple[int, int]] = []
    cx, cy = 880, 315
    radius = 230
    n = 14
    for i in range(n):
        angle = (i / n) * 2 * math.pi + rng.uniform(-0.1, 0.1)
        r = radius * rng.uniform(0.55, 1.0)
        nodes.append((cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle))))

    edge_colors = [ACCENT_GG, ACCENT_TOPIC, ACCENT_KW]
    for i, p in enumerate(nodes):
        # Connect each node to ~2-3 nearby neighbors for a sparse network look.
        dists = sorted(
            ((j, math.hypot(p[0] - q[0], p[1] - q[1])) for j, q in enumerate(nodes) if j != i),
            key=lambda kv: kv[1],
        )[:3]
        for j, _ in dists:
            color = edge_colors[(i + j) % 3]
            # Translucent stroke via additive alpha emulation: blend toward bg.
            draw.line([p, nodes[j]], fill=color, width=2)

    for p in nodes:
        x, y = p
        draw.ellipse((x - 16, y - 16, x + 16, y + 16), fill=NODE, outline=BG, width=3)

    # Subtle gradient panel behind the text on the left so it always reads
    # against the network on the right.
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for x in range(0, 760):
        alpha = int(220 * (1 - x / 760))
        overlay_draw.line([(x, 0), (x, H)], fill=(*BG, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # --- Text block --------------------------------------------------------
    pad_left = 60
    y = 90
    draw_pill(draw, (pad_left, y), "Khoury Research Day · May 4, 2026",
              tag_font, BG, ACCENT_GG)

    y += 78
    draw.text((pad_left, y), "Khoury Researcher", font=title_font, fill=TEXT)
    y += 78
    draw.text((pad_left, y), "Connections", font=title_font, fill=TEXT)

    y += 110
    sub = "Find collaborators across give/get matches,"
    draw.text((pad_left, y), sub, font=sub_font, fill=MUTED)
    y += 38
    draw.text((pad_left, y), "topic similarity, and shared keywords.",
              font=sub_font, fill=MUTED)

    y = H - 70
    draw.text((pad_left, y), "johnguerra.co/viz/khouryResearch",
              font=url_font, fill=ACCENT_GG)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG", optimize=True)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    render()
