# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy"]
# ///
"""Turn the avatar into the ASCII portraits on the card. Run once when the photo changes:

    uv run card/make_ascii.py

It writes ascii_dark.txt (bright skin = dense characters, for light text on a dark
card) and ascii_light.txt (dark hair and jacket = dense characters, for dark text
on a light card, so the face doesn't render as a photo negative).

The photo is dark, turned to the side and sits in front of a brick wall, so a plain
brightness-to-character mapping turns it into one blob. Three steps fix that:
cut the person out of the wall, boost local contrast so the curls and the face
get detail, and use a short ramp whose characters differ clearly in weight.
The numbers below are tuned to this photo; a new photo needs new ones.
"""

from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
ROWS = 44  # matches the card height at the portrait's line height in build.py
CELL_RATIO = 0.48  # monospace cell width / line height in the SVG
CROP = (80, 50, 350, 410)  # x0, y0, x1, y1 on the 460x460 avatar: curls to jacket
PERSON_BOX = (85, 40, 330, 420)  # GrabCut seed: x, y, w, h around the person
# GrabCut drops the curls (they are as dark as the wall), so add them back by hand.
HAIR = {"center": (198, 135), "axes": (102, 72), "angle": -8}
CLAHE_CLIP = 4.0
# Per theme: on the dark card the curls are the dim part, so lift them with a lower gamma.
GAMMA = {"dark": 0.8, "light": 1.2}
RAMP = " .:-=+*#%@"


def cutout(img: np.ndarray) -> np.ndarray:
    """Return a soft 0..1 mask of the person."""
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, PERSON_BOX, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:  # keep only the biggest blob, which is the person
        fg = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    hair = np.zeros_like(fg)
    cv2.ellipse(hair, HAIR["center"], HAIR["axes"], HAIR["angle"], 0, 360, 1, -1)
    fg = np.maximum(fg, hair)
    return cv2.GaussianBlur(fg.astype(np.float32), (7, 7), 0)


def main() -> None:
    img = cv2.imread(str(HERE / "avatar.jpg"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(6, 6)).apply(gray)
    mask = cutout(img)
    person = (gray.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)

    x0, y0, x1, y1 = CROP
    person, mask = person[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(person, (0, 0), 1)
    person = cv2.addWeighted(person, 2.2, blur, -1.2, 0)  # unsharp mask
    cols = round(ROWS * (x1 - x0) / (y1 - y0) / CELL_RATIO)
    small = cv2.resize(person, (cols, ROWS), interpolation=cv2.INTER_AREA) / 255
    inside = cv2.resize(mask, (cols, ROWS), interpolation=cv2.INTER_AREA) > 0.6

    for theme in ("dark", "light"):
        lines = []
        for row, row_inside in zip(small, inside, strict=True):
            line = ""
            for v, keep in zip(row, row_inside, strict=True):
                v = v ** GAMMA[theme]
                if theme == "light":
                    v = 1 - v if keep else 0.0
                line += " " if v <= 0.04 else RAMP[min(len(RAMP) - 1, int(v * len(RAMP)))]
            lines.append(line.rstrip())
        (HERE / f"ascii_{theme}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
