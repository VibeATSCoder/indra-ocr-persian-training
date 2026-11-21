"""Card finder: a tiny segmentation net that marks the card on a page, exported as card.onnx. Kaggle GPU.

Shape rules find 93-100% of scanned cards but 25% of cards held in a hand, so this learns from MIDV-UP's photos and scans with
known corners, in any rotation and zoom, up to a card that fills the frame; templates pasted on other backgrounds add variety.
Test cards never reach this script. Input: a grayscale page, long side 256, padded square. Output: card logit map, same size.

DRY_RUN=1 runs on CPU with whatever the zip holds (local check).
MIDV-UP: Chernyshova, Ilyukhin, Arlazarov, ICDAR 2025, CC BY-SA 2.5.
"""
import collections
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import midv

DRY = os.environ.get("DRY_RUN") == "1"
OUT = Path(os.environ.get("KAGGLE_OUT", "/kaggle/working"))
ZIP = Path(os.environ.get("MIDV_ZIP", "/tmp/midv-up.zip"))
SIZE = 256
BATCH = 4 if DRY else 48
TRAIN_SECONDS = float(os.environ.get("TRAIN_SECONDS", 40 if DRY else 1800))
EVAL_EVERY = 15 if DRY else 240
SEED = 20261124
DOCS = ("irn_id", "irn_drvlic", "irn_bc", "pak_id")  # all card-shaped or nearly; the net must find the object, not read it
ALL_TRAIN = os.environ.get("CARD_ALL") == "1"  # local check on validation cards only
cv2.setNumThreads(1)
torch.set_num_threads(2)


class Block(nn.Module):
    def __init__(self, a, b, stride=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(a, b, 3, stride, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True), nn.Conv2d(b, b, 3, 1, 1, groups=b, bias=False), nn.Conv2d(b, b, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class CardNet(nn.Module):
    def __init__(self, c=(12, 24, 40, 64, 96)):
        super().__init__()
        self.down = nn.ModuleList([Block(1, c[0], 2)] + [Block(c[i], c[i + 1], 2) for i in range(4)])
        self.up = nn.ModuleList([Block(c[i + 1] + c[i], c[i]) for i in range(3, -1, -1)])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for d in self.down:
            x = d(x)
            skips.append(x)
        x = skips.pop()
        for u in self.up:
            s = skips.pop()
            x = u(torch.cat([F.interpolate(x, size=s.shape[2:], mode="nearest"), s], 1))
        return F.interpolate(self.head(x), scale_factor=2, mode="bilinear", align_corners=False)


def letterbox_matrix(w, h, size=SIZE):
    s = size / max(w, h)
    return np.float32([[s, 0, (size - w * s) / 2], [0, s, (size - h * s) / 2]])


def load_views():
    """Pages at most 640 px on the long side, with the card's corners on them. Test cards are not opened."""
    views, templates = {"train": [], "validation": []}, []
    with zipfile.ZipFile(ZIP) as z:
        names = set(z.namelist())
        meta = {}

        def ann(path):
            if path not in meta:
                meta[path] = json.load(z.open(path)) if path in names else {"_via_img_metadata": {}}
            return meta[path]

        for doc in DOCS:
            for i in range(1, 251):
                card = f"{i:03d}"
                split = midv.split_of(card)
                assert card not in midv.TEST or split == "test"
                if split == "test":
                    continue
                if ALL_TRAIN:
                    split = "train" if i % 3 else "validation"
                for view, member, annotation, fname in midv.views(card, doc, []):
                    if member not in names:
                        continue
                    raw = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_GRAYSCALE)
                    if raw is None:
                        continue
                    h, w = raw.shape
                    s = min(1.0, 640 / max(h, w))
                    small = cv2.resize(raw, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else raw
                    if view == "template":
                        if split == "train":
                            templates.append(small)
                        continue
                    quad = next((np.float32(list(zip(r["shape_attributes"]["all_points_x"], r["shape_attributes"]["all_points_y"]))) * s for r in midv.regions(ann(annotation), fname) if r["region_attributes"].get("field_name") == "doc_quad"), None)
                    if quad is not None:
                        views[split].append({"image": small, "quad": quad, "view": view, "doc": doc})
                if i % 50 == 0:
                    print("LOADED", doc, i, {k: len(v) for k, v in views.items()}, flush=True)
    return views, templates


def background(rs, photos):
    """A plain, textured or blurred-photo background with no card on it."""
    size = int(rs.integers(320, 640)), int(rs.integers(320, 640))
    kind = rs.random()
    if kind < .35 and photos:
        p = photos[int(rs.integers(len(photos)))]
        m = np.zeros(p["image"].shape, np.uint8)
        cv2.fillConvexPoly(m, np.int32(p["quad"]), 1)
        fill = cv2.inpaint(cv2.resize(p["image"], None, fx=.25, fy=.25), cv2.resize(m, None, fx=.25, fy=.25, interpolation=cv2.INTER_NEAREST), 5, cv2.INPAINT_TELEA)
        return cv2.resize(fill, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    base = np.full(size, float(rs.integers(10, 250)), np.float32)
    base += cv2.resize(rs.normal(0, rs.uniform(2, 45), (rs.integers(2, 24), rs.integers(2, 24))).astype(np.float32), (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    if kind > .7:  # wood grain or fabric: streaks
        streak = cv2.GaussianBlur(rs.normal(0, 30, size).astype(np.float32), (0, 0), sigmaX=float(rs.uniform(10, 40)), sigmaY=1)
        base += streak if rs.random() < .5 else streak.T[: size[0], : size[1]] if size[0] == size[1] else streak
    return np.clip(base + rs.normal(0, rs.uniform(0, 6), size), 0, 255).astype(np.uint8)


def compose(template, rs, photos):
    bg = background(rs, photos)
    h, w = bg.shape
    th, tw = template.shape
    s = rs.uniform(.25, .95) * min(w / tw, h / th)
    a, cx, cy = rs.uniform(-180, 180), rs.uniform(.25, .75) * w, rs.uniform(.25, .75) * h
    M = cv2.getRotationMatrix2D((tw / 2, th / 2), a, s)
    M[:, 2] += (cx - tw / 2, cy - th / 2)
    card = cv2.warpAffine(template, M, (w, h), flags=cv2.INTER_AREA)
    mask = cv2.warpAffine(np.full(template.shape, 255, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST)
    out = np.where(mask > 0, card, bg)
    quad = cv2.transform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])[None], M)[0]
    return out.astype(np.uint8), quad


def jitter(a, rs):
    x = a.astype(np.float32)
    if rs.random() < .6:
        x = x * rs.uniform(.45, 1.25) + rs.uniform(-35, 35)
    if rs.random() < .3:
        x = x + (np.linspace(rs.uniform(-40, 40), rs.uniform(-40, 40), x.shape[1])[None, :] + np.linspace(rs.uniform(-40, 40), rs.uniform(-40, 40), x.shape[0])[:, None])
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rs.random() < .3:
        x = cv2.GaussianBlur(x, (0, 0), rs.uniform(.5, 2))
    if rs.random() < .4:
        x = np.clip(x + rs.normal(0, rs.uniform(2, 20), x.shape), 0, 255).astype(np.uint8)
    return x


class Pages(Dataset):
    def __init__(self, views, templates, n):
        self.views, self.templates, self.n, self.count = views, templates, n, 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self.count += 1
        rs = np.random.default_rng((torch.initial_seed() % (1 << 62), self.count))
        if self.templates and rs.random() < .3:
            a, quad = compose(self.templates[int(rs.integers(len(self.templates)))], rs, self.views)
        else:
            v = self.views[int(rs.integers(len(self.views)))]
            a, quad = v["image"], v["quad"]
        h, w = a.shape
        # A random view around the card: from a far shot down to a card that fills the frame.
        q0, q1 = quad.min(0), quad.max(0)
        c, half = (q0 + q1) / 2, (q1 - q0) / 2
        if rs.random() < .45:
            far = 1 + rs.uniform(0, .3) if rs.random() < .5 else rs.uniform(1.3, 2.5)
            hx, hy = half[0] * far + 2, half[1] * far + 2
            x0, y0 = int(max(0, c[0] - hx + rs.uniform(-.1, .1) * hx)), int(max(0, c[1] - hy + rs.uniform(-.1, .1) * hy))
            x1, y1 = int(min(w, c[0] + hx + rs.uniform(-.1, .1) * hx)), int(min(h, c[1] + hy + rs.uniform(-.1, .1) * hy))
            if x1 - x0 > 32 and y1 - y0 > 32:
                a, quad = a[y0:y1, x0:x1], quad - (x0, y0)
                h, w = a.shape
        turn = int(rs.integers(4))
        for _ in range(turn):  # quarter turns are common and keep every pixel
            a = np.rot90(a)
            quad = np.stack([quad[:, 1], w - quad[:, 0]], 1)
            h, w = w, h
        a = np.ascontiguousarray(a)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rs.uniform(-30, 30) if rs.random() < .5 else 0, rs.uniform(.9, 1.1))
        a = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        quad = cv2.transform(quad[None], M)[0]
        L = letterbox_matrix(w, h)
        small = cv2.warpAffine(jitter(a, rs), L, (SIZE, SIZE), flags=cv2.INTER_AREA, borderValue=255)
        quad = cv2.transform(quad[None], L)[0]
        mask = np.zeros((SIZE, SIZE), np.float32)
        cv2.fillConvexPoly(mask, np.int32(quad), 1.0)
        return torch.from_numpy(1 - small.astype(np.float32) / 255)[None], torch.from_numpy(mask)


def evaluate(model, pages, device):
    """Median IoU and share with IoU >= .8, per view, on unchanged validation pages."""
    model.eval()
    out = collections.defaultdict(list)
    with torch.inference_mode():
        for v in pages:
            h, w = v["image"].shape
            L = letterbox_matrix(w, h)
            small = cv2.warpAffine(v["image"], L, (SIZE, SIZE), flags=cv2.INTER_AREA, borderValue=255)
            q = cv2.transform(v["quad"][None], L)[0]
            truth = np.zeros((SIZE, SIZE), np.uint8)
            cv2.fillConvexPoly(truth, np.int32(q), 1)
            pred = (model(torch.from_numpy(1 - small.astype(np.float32) / 255)[None, None].to(device))[0, 0].cpu().numpy() > 0).astype(np.uint8)
            out[v["view"]].append((pred & truth).sum() / max(1, (pred | truth).sum()))
    return {k: {"median": float(np.median(x)), "ok": float((np.array(x) >= .8).mean()), "n": len(x)} for k, x in out.items()}


def main():
    t0 = time.monotonic()
    if not ZIP.exists():
        proc = subprocess.Popen(["curl", "-s", "-C", "-", "--retry", "30", "--retry-delay", "20", "--retry-all-errors", midv.URL, "-o", str(ZIP)])
        while proc.poll() is None:
            time.sleep(60)
            print("DOWNLOAD", round(ZIP.stat().st_size / 1e9, 2) if ZIP.exists() else 0, "GB", flush=True)
        assert proc.returncode == 0 and ZIP.stat().st_size == midv.SIZE, "download incomplete"
    views, templates = load_views()
    print("PAGES", {k: collections.Counter(x["view"] for x in v) for k, v in views.items()}, "templates", len(templates), round(time.monotonic() - t0), "s", flush=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not DRY else "cpu")
    model = CardNet().to(device)
    print("PARAMS", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    data = DataLoader(Pages(views["train"], templates, 10 ** 7), BATCH, shuffle=False, num_workers=0 if DRY else 3, persistent_workers=not DRY, pin_memory=not DRY, drop_last=True)
    start = last = time.monotonic()
    best, steps, log = -1.0, 0, []
    for x, y in data:
        frac = (time.monotonic() - start) / TRAIN_SECONDS
        for g in opt.param_groups:
            g["lr"] = 2e-3 * (.02 + .98 * .5 * (1 + math.cos(math.pi * min(1, frac))))
        model.train()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logit = model(x)[:, 0]
        p = torch.sigmoid(logit)
        loss = F.binary_cross_entropy_with_logits(logit, y) + 1 - ((2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps += 1
        now = time.monotonic()
        final = now - start > TRAIN_SECONDS
        if now - last > EVAL_EVERY or final:
            m = evaluate(model, views["validation"], device)
            value = float(np.mean([v["ok"] + v["median"] for v in m.values()]))
            log.append({"steps": steps, "seconds": round(now - start), "loss": round(loss.item(), 4), "score": value, "views": m})
            print("VALID", json.dumps(log[-1]), flush=True)
            if value > best:
                best = value
                torch.save(model.state_dict(), OUT / "card.pt")
            last = time.monotonic()
        if final:
            break
    model = CardNet()
    model.load_state_dict(torch.load(OUT / "card.pt", map_location="cpu"))
    model.eval()
    x = torch.rand(1, 1, SIZE, SIZE)
    torch.onnx.export(model, x, OUT / "card.onnx", input_names=["x"], output_names=["mask"], opset_version=17, dynamo=False)
    import onnxruntime as ort

    a = ort.InferenceSession(str(OUT / "card.onnx"), providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
    with torch.inference_mode():
        parity = float(np.abs(a - model(x).numpy()).max())
    assert parity < 1e-3, parity
    (OUT / "card-results.json").write_text(json.dumps({"best": best, "parity": parity, "log": log, "source": "MIDV-UP, CC BY-SA 2.5"}, indent=1), encoding="utf8")
    print("FINAL", json.dumps({"best": best, "parity": parity, "steps": steps, "bytes": (OUT / "card.onnx").stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
