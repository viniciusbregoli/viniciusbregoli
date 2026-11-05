# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow"]
# ///
"""Turn the avatar into the ASCII portrait on the card. Run once when the photo changes:

    uv run card/make_ascii.py > card/ascii.txt
"""
from pathlib import Path

from PIL import Image, ImageFilter

HERE = Path(__file__).parent
COLS = 50
CROP = (85, 45, 335, 320)  # head and shoulders of the 460x460 avatar
# Levels tuned to this photo: the brick wall sits around 22, hair around 43,
# the face around 170-200. Clipping at 26 hides the wall but keeps the curls.
LOW, HIGH, GAMMA = 26, 210, 0.75
# A monospace cell at 16px with 20px line height is about half as wide as tall.
CELL_RATIO = 0.48
RAMP = " .`',:;!ilI|1j(fr/tvzcunxJYUL0QOZmwqpdbkhao*#MW&8%B@"

img = Image.open(HERE / "avatar.jpg").convert("L").crop(CROP)
img = img.filter(ImageFilter.UnsharpMask(2, 160, 2))
w, h = img.size
rows = round(COLS * h / w * CELL_RATIO)
img = img.resize((COLS, rows), Image.LANCZOS)

for y in range(rows):
    line = ""
    for x in range(COLS):
        v = (img.getpixel((x, y)) - LOW) / (HIGH - LOW)
        v = max(0.0, min(1.0, v)) ** GAMMA
        line += RAMP[min(len(RAMP) - 1, int(v * len(RAMP)))]
    print(line.rstrip())
