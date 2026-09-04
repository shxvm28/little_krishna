import cv2
import numpy as np
import random

INPUT = "krishna.png"
OUTPUT = "neon_krishna.mp4"

W, H = 1920, 1080
FPS = 30

DOT_TIME = 3.5
REFINE_TIME = 3.0
NEON_HOLD = 3.0
FADE_TIME = 1.5
FINAL_HOLD = 3.0

BLOCK = 10


def fit_image():
    img = cv2.imread(INPUT)

    if img is None:
        raise FileNotFoundError("krishna.png not found")

    h, w = img.shape[:2]
    scale = min(W / w, H / h)

    nw = int(w * scale)
    nh = int(h * scale)

    img = cv2.resize(img, (nw, nh))

    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    x = (W - nw) // 2
    y = (H - nh) // 2

    canvas[y:y+nh, x:x+nw] = img

    return canvas


def make_neon(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(
        gray, 7, 50, 50
    )

    edges = cv2.Canny(
        gray, 60, 150
    )

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), np.uint8),
        iterations=2
    )

    hsv = cv2.cvtColor(
        img, cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * 1.8,
        0, 255
    )

    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2] * 1.6 + 50,
        0, 255
    )

    boosted = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR
    )

    edge = np.zeros_like(img)
    edge[edges > 0] = boosted[edges > 0]

    glow1 = cv2.GaussianBlur(
        edge, (0, 0), 5
    )

    glow2 = cv2.GaussianBlur(
        edge, (0, 0), 20
    )

    neon = (
        edge.astype(np.float32) * 1.5
        + glow1.astype(np.float32) * 1.2
        + glow2.astype(np.float32) * 0.9
    )

    return np.clip(
        neon, 0, 255
    ).astype(np.uint8)


def add_particles(frame, amount, seed):

    layer = np.zeros_like(frame)

    random.seed(seed)

    for _ in range(int(900 * amount)):

        x = random.randint(50, W - 50)
        y = random.randint(50, H - 50)

        r = random.choice([1, 1, 2, 2, 3])

        if random.random() < 0.5:
            color = (
                255,
                random.randint(160, 255),
                255
            )
        else:
            color = (
                random.randint(180, 255),
                40,
                255
            )

        cv2.circle(
            layer,
            (x, y),
            r,
            color,
            -1,
            cv2.LINE_AA
        )

    glow = cv2.GaussianBlur(
        layer, (0, 0), 6
    )

    return cv2.addWeighted(
        frame, 1,
        glow, 1,
        0
    )


def mosaic(img, size):

    if size <= 1:
        return img.copy()

    small = cv2.resize(
        img,
        (W // size, H // size),
        interpolation=cv2.INTER_AREA
    )

    return cv2.resize(
        small,
        (W, H),
        interpolation=cv2.INTER_NEAREST
    )


def main():

    print("Loading Krishna...")

    original = fit_image()

    print("Creating neon artwork...")

    neon = make_neon(original)

    codec = cv2.VideoWriter_fourcc(*"mp4v")

    video = cv2.VideoWriter(
        OUTPUT,
        codec,
        FPS,
        (W, H)
    )

    # -----------------------------
    # DOT REVEAL
    # -----------------------------

    print("Creating dot reveal...")

    small = cv2.resize(
        neon,
        (W // BLOCK, H // BLOCK),
        interpolation=cv2.INTER_AREA
    )

    positions = [
        (y, x)
        for y in range(small.shape[0])
        for x in range(small.shape[1])
    ]

    random.seed(42)
    random.shuffle(positions)

    canvas = np.zeros_like(small)

    frames = int(DOT_TIME * FPS)

    for i in range(frames):

        start = int(
            len(positions) * i / frames
        )

        end = int(
            len(positions) * (i + 1) / frames
        )

        for y, x in positions[start:end]:

            if np.max(small[y, x]) > 15:
                canvas[y, x] = small[y, x]

        frame = cv2.resize(
            canvas,
            (W, H),
            interpolation=cv2.INTER_NEAREST
        )

        frame = add_particles(
            frame,
            (i + 1) / frames,
            i
        )

        video.write(frame)

    # -----------------------------
    # REFINEMENT
    # -----------------------------

    print("Refining neon drawing...")

    frames = int(REFINE_TIME * FPS)

    for i in range(frames):

        t = (i + 1) / frames

        size = max(
            1,
            int(BLOCK * (1 - t) ** 2)
        )

        frame = mosaic(neon, size)

        frame = add_particles(
            frame,
            0.7,
            i + 1000
        )

        video.write(frame)

    # -----------------------------
    # NEON HOLD
    # -----------------------------

    print("Holding neon artwork...")

    frames = int(NEON_HOLD * FPS)

    for i in range(frames):

        frame = add_particles(
            neon.copy(),
            0.8,
            i + 2000
        )

        video.write(frame)

    # -----------------------------
    # CROSSFADE
    # -----------------------------

    print("Crossfading to original...")

    frames = int(FADE_TIME * FPS)

    for i in range(frames):

        t = (i + 1) / frames

        frame = cv2.addWeighted(
            neon,
            1 - t,
            original,
            t,
            0
        )

        video.write(frame)

    # -----------------------------
    # FINAL HOLD
    # -----------------------------

    print("Final frame...")

    frames = int(FINAL_HOLD * FPS)

    for _ in range(frames):
        video.write(original)

    video.release()

    print("DONE!")
    print("neon_krishna.mp4 created")


if __name__ == "__main__":
    main()
