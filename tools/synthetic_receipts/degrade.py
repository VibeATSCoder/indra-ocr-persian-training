import math
import cv2
import numpy as np

LEVELS = {
    "light": (0.75, 5, 60),
    "medium": (0.5, 12, 40),
    "heavy": (0.35, 20, 25),
    "extreme": (0.25, 30, 15),
}


def lower_quality(image, scale, noise, quality, rs):
    """image: uint8 grayscale or BGR. Returns the re-encoded image, as a messenger or a cheap camera leaves it."""
    h, w = image.shape[:2]
    small = cv2.resize(
        image, (max(8, round(w * scale)), max(8, round(h * scale))), interpolation=cv2.INTER_AREA
    )
    grain = rs.normal(0, noise, small.shape[:2]).astype(np.float32)
    noisy = small.astype(np.float32) + (grain[..., None] if small.ndim == 3 else grain)
    speck = rs.random(small.shape[:2]) < noise / 1800
    noisy[speck] = (
        rs.choice([0, 255], int(speck.sum()))[..., None]
        if small.ndim == 3
        else rs.choice([0, 255], int(speck.sum()))
    )
    ok, buf = cv2.imencode(
        ".jpg", np.clip(noisy, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def transform(image, lines, level, rs, rotate=0, perspective=0):
    h, w = image.shape[:2]
    H = np.eye(3)
    paper = tuple((int(x) for x in np.median(image.reshape(-1, 3)[::50], axis=0)))
    if rotate:
        angle = float(rs.uniform(-rotate, rotate))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        nw = math.ceil(h * abs(M[0, 1]) + w * abs(M[0, 0]))
        nh = math.ceil(h * abs(M[0, 0]) + w * abs(M[0, 1]))
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
        image = cv2.warpAffine(image, M, (nw, nh), borderValue=paper)
        H = np.vstack([M, [0, 0, 1]]) @ H
        h, w = image.shape[:2]
    if perspective:
        margin = math.ceil(max(h, w) * perspective) + 2
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = (
            src
            + margin
            + rs.uniform(-min(h, w) * perspective, min(h, w) * perspective, (4, 2)).astype(
                np.float32
            )
        )
        M = cv2.getPerspectiveTransform(src, dst)
        image = cv2.warpPerspective(image, M, (w + 2 * margin, h + 2 * margin), borderValue=paper)
        H = M @ H
    before = image.shape[:2]
    if level in LEVELS:
        image = lower_quality(image, *LEVELS[level], rs)
    elif level == "photo":
        h, w = image.shape[:2]
        a = (
            image.astype(np.float32)
            + np.linspace(-45, 15, w)[None, :, None]
            + np.linspace(-20, 10, h)[:, None, None]
        )
        a += rs.normal(0, 9, (h, w, 1))
        a = cv2.GaussianBlur(np.clip(a, 0, 255).astype(np.uint8), (0, 0), 1.25)
        image = cv2.imdecode(cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 55])[1], 1)
    elif level != "clean":
        raise ValueError("Unknown quality level: " + level)
    after = image.shape[:2]
    H = np.diag([after[1] / before[1], after[0] / before[0], 1]) @ H
    for line in lines:
        line["source_bbox"] = line["bbox"][:]
        points = np.asarray(line["polygon"], np.float64)
        points = np.c_[points, np.ones(4)] @ H.T
        points = points[:, :2] / points[:, 2:]
        line["polygon"] = np.round(points, 3).tolist()
        line["bbox"] = [
            max(0, math.floor(points[:, 0].min())),
            max(0, math.floor(points[:, 1].min())),
            min(after[1], math.ceil(points[:, 0].max())),
            min(after[0], math.ceil(points[:, 1].max())),
        ]
    return (image, lines, H.tolist())
