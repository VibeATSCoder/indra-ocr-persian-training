"""Detector fine-tune: on a damaged page, do what the original detector does on the clean page.

Every step shows the frozen original detector a clean page and the student a damaged copy of it, and the student learns
the teacher's output: line core, top and bottom distance. So its behaviour on clean pages is the reference, and what a
line is stays as Indra defines it. Damage follows lowq.py's chain (fewer pixels, noise, heavy JPEG) with blur, low
contrast, uneven light and blotches.

Data: synthetic pages (font families disjoint per split) and real receipts, train and validation only; held-out test
receipts never reach Kaggle and are scored locally through the whole pipeline.
Selection: box agreement with the teacher's clean boxes at IoU .5, .7 and .85, on validation pages at lowq.py's levels.
Exact box position matters: with hand-checked boxes the held-out fields rise far more than recall alone explains.
The original detector is an eligible candidate.

DRY_RUN=1 runs the whole pipeline on CPU with a tiny subset (local check, no Kaggle).
"""
import collections
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from det_post import INPUT_MULTIPLE, decode, page_to_input
from det_trainable import DetNet, verify_parity
from lowq import LEVELS, lower_quality

DRY = os.environ.get("DRY_RUN") == "1"
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("KAGGLE_OUT", "/kaggle/working"))
TMP = Path(os.environ.get("TRAIN_TMP", "/tmp/det-train"))
TMP.mkdir(parents=True, exist_ok=True)
SEED = 20261120
LONG_SIDE = 1024  # meta.json det_long_side
BATCH = 2 if DRY else 8
TRAIN_SECONDS = float(os.environ.get("TRAIN_SECONDS", 40 if DRY else 2400))  # per candidate
EVAL_EVERY = 20 if DRY else 400
REAL_SHARE = .4  # of training samples
VALID_SYNTHETIC = 4 if DRY else 120
MODEL_REV = "bd02167fdf2005a19962cc15188f8496a2c87d3b"
EVAL_LEVELS = {"original": .25, "light": .3, "medium": .3, "heavy": .15}  # selection weights; extreme is shown, not selected on
IOUS = (.5, .7, .85)
CONFIGS = [{"name": "steady", "lr": 1e-4, "anchor": 1e-5, "pos": 1.0},
           {"name": "recall", "lr": 1e-4, "anchor": 1e-5, "pos": 2.0},
           {"name": "bold", "lr": 3e-4, "anchor": 0.0, "pos": 1.0},
           {"name": "gentle", "lr": 3e-5, "anchor": 1e-5, "pos": 1.0}]
assert not (set(p.name for p in INPUT.iterdir()) & {"receipts-eval"}) and next(INPUT.rglob("receipts-eval.tar"), None) is None and next(INPUT.rglob("labels-r2"), None) is None, "the sealed evaluation set is mounted: never train with it"
cv2.setNumThreads(1)
torch.set_num_threads(2)


def start_model():
    found = os.environ.get("DET_START")
    if found:
        return Path(found)
    path = TMP / "det.onnx"
    if not path.exists():
        r = requests.get(f"https://huggingface.co/datalab-to/indra-ocr/resolve/{MODEL_REV}/det.onnx", timeout=180)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


ORIGINAL = start_model()
print("START", ORIGINAL, hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()[:16], flush=True)

# ------------------------------------------------------------------ pages
t0 = time.monotonic()
real_path = next(INPUT.rglob("real-pages.json"))
real = json.loads(real_path.read_text(encoding="utf8"))
synth_path = next(INPUT.rglob("synthetic-pages.json"))
synth = json.loads(synth_path.read_text(encoding="utf8"))
PAGES = TMP / "pages"
PAGES.mkdir(exist_ok=True)
if not any(PAGES.iterdir()):  # the input disk can be slow on Kaggle: work from a local copy
    with tarfile.open(next(INPUT.rglob("synthetic-pages.tar"))) as tar:
        tar.extractall(PAGES, filter="data")
    print("CACHE synthetic", round(time.monotonic() - t0), "s", flush=True)
    for row in real:
        shutil.copyfile(real_path.parent / row["file"], PAGES / row["file"])
assert all(r["split"] in ("train", "validation") for r in real), "a held-out receipt reached training"
assert not {r["font"] for r in synth if r["split"] == "train"} & {r["font"] for r in synth if r["split"] == "validation"}
ITEMS = {s: [{"path": PAGES / r["file"], "skew": r["skew"], "source": "real"} for r in real if r["split"] == s]
         + [{"path": PAGES / r["file"], "skew": 0.0, "source": "synthetic"} for r in synth if r["split"] == s] for s in ("train", "validation")}
if DRY:
    ITEMS["train"] = ITEMS["train"][:8] + ITEMS["train"][-8:]
print("PAGES", {s: dict(collections.Counter(i["source"] for i in v)) for s, v in ITEMS.items()}, round(time.monotonic() - t0), "s", flush=True)


def read_page(item):
    """BGR page, straightened as the pipeline straightens it."""
    a = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
    if abs(item["skew"]) >= .3:  # pipeline.DESKEW_MIN
        h, w = a.shape[:2]
        a = cv2.warpAffine(a, cv2.getRotationMatrix2D((w / 2, h / 2), item["skew"], 1.0), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return a


def to_gray(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)


def to_input(gray, nh, nw):
    """page_to_input at a size already chosen, so the damaged page lines up with the clean one."""
    h, w = gray.shape
    if (h, w) != (nh, nw):
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA if nh < h else cv2.INTER_LINEAR)
    out = np.zeros((math.ceil(nh / INPUT_MULTIPLE) * INPUT_MULTIPLE, math.ceil(nw / INPUT_MULTIPLE) * INPUT_MULTIPLE), np.float32)
    out[:nh, :nw] = 1.0 - gray.astype(np.float32) / 255.0
    return out


def damage(a, rs):
    """A random page damage. Severity is drawn once so size, noise and compression go together, as in lowq.LEVELS:
    severity .25, .5, .75 and 1 are close to light, medium, heavy and extreme."""
    x = a.astype(np.float32)
    h, w = a.shape[:2]
    if rs.random() < .15:  # low contrast
        x = 128 + (x - 128) * rs.uniform(.5, .9) + rs.uniform(-30, 30)
    if rs.random() < .12:  # uneven light
        x = x + (np.linspace(rs.uniform(-45, 15), rs.uniform(-45, 15), w)[None, :] + np.linspace(rs.uniform(-25, 15), rs.uniform(-25, 15), h)[:, None])[..., None]
    if rs.random() < .08:  # blotchy paper
        blotch = cv2.resize(rs.normal(0, 1, (max(2, h // 40), max(2, w // 40))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        x = x + rs.uniform(6, 18) * blotch[..., None]
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rs.random() < .2:  # soft focus
        x = cv2.GaussianBlur(x, (0, 0), rs.uniform(.5, 1.5))
    severity = rs.random() ** 1.2
    scale = float(np.clip(.25 ** severity * rs.uniform(.85, 1.15), .2, 1))
    noise = 30 * severity * rs.uniform(.5, 1.3)
    quality = int(np.clip((80 - 65 * severity) * rs.uniform(.8, 1.25), 10, 95))
    return lower_quality(x, scale, noise, quality, rs)


class Pages(Dataset):
    """Clean input, damaged input and the page's size in the input."""

    def __init__(self, items):
        self.items, self.count = items, itertools.count()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rs = np.random.default_rng((torch.initial_seed() % (1 << 62), next(self.count)))
        a = read_page(self.items[i])
        if rs.random() < .35:  # people crop their screenshots: text comes larger in the detector's input
            h, w = a.shape[:2]
            ch, cw = int(h * rs.uniform(.55, 1)), int(w * rs.uniform(.8, 1))
            y, x = int(rs.integers(0, h - ch + 1)), int(rs.integers(0, w - cw + 1))
            a = a[y:y + ch, x:x + cw]
        clean, s = page_to_input(to_gray(a), LONG_SIDE)
        h, w = a.shape[:2]
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        bad = clean if rs.random() < .1 else to_input(to_gray(damage(a, rs)), nh, nw)
        return clean, bad, nh, nw


def collate(batch):
    H, W = max(b[0].shape[0] for b in batch), max(b[0].shape[1] for b in batch)
    clean, bad, valid = np.zeros((len(batch), 1, H, W), np.float32), np.zeros((len(batch), 1, H, W), np.float32), np.zeros((len(batch), H // 2, W // 2), np.float32)
    for i, (c, d, nh, nw) in enumerate(batch):
        clean[i, 0, :c.shape[0], :c.shape[1]] = c
        bad[i, 0, :d.shape[0], :d.shape[1]] = d
        valid[i, :math.ceil(nh / 2), :math.ceil(nw / 2)] = 1
    return torch.from_numpy(clean), torch.from_numpy(bad), torch.from_numpy(valid)


def train_loader(seed):
    items = ITEMS["train"]
    n = collections.Counter(i["source"] for i in items)
    p = [(REAL_SHARE if i["source"] == "real" else 1 - REAL_SHARE) / n[i["source"]] for i in items]
    sampler = WeightedRandomSampler(p, 1_000_000, replacement=True, generator=torch.Generator().manual_seed(seed))
    return DataLoader(Pages(items), BATCH, sampler=sampler, collate_fn=collate, num_workers=0 if DRY else 2, pin_memory=not DRY, persistent_workers=not DRY, drop_last=True)


def objective(s, t, valid, pos):
    """Student s against teacher t, both [B, 3, h, w]: line core (BCE and Dice), then top and bottom distance relative to line height."""
    p = torch.sigmoid(t[:, 0])
    bce = (F.binary_cross_entropy_with_logits(s[:, 0], p, reduction="none") * valid * (1 + (pos - 1) * p)).sum() / valid.sum().clamp_min(1)
    q = torch.sigmoid(s[:, 0]) * valid
    dice = 1 - ((2 * (q * p).sum((1, 2)) + 1) / (q.sum((1, 2)) + (p * valid).sum((1, 2)) + 1)).mean()
    core = (p > .5) & (valid > 0)
    height = (t[:, 1] + t[:, 2]).clamp_min(2)
    off = F.smooth_l1_loss(s[:, 1] / height, t[:, 1] / height, reduction="none", beta=.1) + F.smooth_l1_loss(s[:, 2] / height, t[:, 2] / height, reduction="none", beta=.1)
    box = off[core].mean() if core.any() else off.sum() * 0
    return bce, dice, box


# ------------------------------------------------------------------ validation: fixed damage at lowq.py's levels
def validation_pages(teacher, device):
    picks = [i for i in ITEMS["validation"] if i["source"] == "real"] + [i for i in ITEMS["validation"] if i["source"] == "synthetic"][:VALID_SYNTHETIC]
    out = []
    for item in picks:
        a = read_page(item)
        clean, s = page_to_input(to_gray(a), LONG_SIDE)
        h, w = a.shape[:2]
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        inputs = {"original": clean}
        rs_seed = int(hashlib.sha256(item["path"].name.encode()).hexdigest()[:8], 16)
        for level in ("light", "medium", "heavy", "extreme"):
            bad = lower_quality(a, *LEVELS[level], np.random.default_rng((rs_seed, list(LEVELS).index(level))))
            inputs[level] = to_input(to_gray(bad), nh, nw)
        with torch.inference_mode():
            ref = decode(teacher(torch.from_numpy(clean)[None, None].to(device))[0].float().cpu().numpy())
        out.append({"source": item["source"], "inputs": inputs, "ref": ref})
    return out


def iou_matrix(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-6)


def matches(pred, ref, thr):
    if not len(pred) or not len(ref):
        return 0
    m = iou_matrix(ref, pred)
    n = 0
    for i in np.argsort(-m.max(1)):
        j = int(m[i].argmax())
        if m[i, j] >= thr:
            n += 1
            m[:, j] = -1
    return n


@torch.inference_mode()
def evaluate(model, pages, device):
    model.eval()
    tally = collections.Counter()
    for page in pages:
        for level, x in page["inputs"].items():
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(torch.from_numpy(x)[None, None].to(device))
            pred = decode(out[0].float().cpu().numpy())
            for thr in IOUS:
                key = (page["source"], level, thr)
                tally[key + ("tp",)] += matches(pred, page["ref"], thr)
                tally[key + ("n",)] += len(pred) + len(page["ref"])
    return {f"{source}/{level}": float(np.mean([2 * tally[source, level, thr, "tp"] / max(1, tally[source, level, thr, "n"]) for thr in IOUS]))
            for source in ("real", "synthetic") for level in page["inputs"]}


def score(metrics):
    return sum(w * .5 * (metrics[f"real/{level}"] + metrics[f"synthetic/{level}"]) for level, w in EVAL_LEVELS.items())


def candidate(rank, config, queue):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and not DRY else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed = SEED + 1000 * [c["name"] for c in CONFIGS].index(config["name"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    teacher = DetNet(ORIGINAL).to(device).eval().requires_grad_(False)
    pages = validation_pages(teacher, device)
    if not (OUT / "baseline.json").exists():  # the start model's own scores: eligible for selection
        base = evaluate(teacher, pages, device)
        (OUT / "baseline.json").write_text(json.dumps(base, indent=1))
        print("BASELINE", round(score(base), 5), json.dumps({k: round(v, 3) for k, v in base.items()}), flush=True)
    student = DetNet(ORIGINAL).to(device)
    anchors = {n: p.detach().clone() for n, p in student.named_parameters()}
    opt = torch.optim.AdamW(student.parameters(), lr=config["lr"], weight_decay=0)

    def lr_at(frac):  # time-based: 3% warmup, cosine to lr/50 exactly at TRAIN_SECONDS
        if frac < .03:
            return config["lr"] * (.1 + .9 * frac / .03)
        return config["lr"] * (.02 + .98 * .5 * (1 + math.cos(math.pi * min(1, (frac - .03) / .97))))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    data = train_loader(seed)
    start = last_eval = time.monotonic()
    best, steps, run, log, done = -1.0, 0, collections.Counter(), [], False
    while not done:
        for clean, bad, valid in data:
            student.train()
            clean, bad, valid = clean.to(device, non_blocking=True), bad.to(device, non_blocking=True), valid.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                s = student(bad)
                with torch.no_grad():
                    t = teacher(clean)
            bce, dice, box = objective(s.float(), t.float(), valid, config["pos"])
            anchor = sum((p - anchors[n]).square().sum() for n, p in student.named_parameters())
            loss = bce + dice + box + config["anchor"] * anchor
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                run["nonfinite_skipped"] += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
            for g in opt.param_groups:
                g["lr"] = lr_at((time.monotonic() - start) / TRAIN_SECONDS)
            scaler.step(opt)
            scaler.update()
            steps += 1
            run.update({"loss": loss.item(), "bce": bce.item(), "dice": dice.item(), "box": box.item(), "anchor": anchor.item()})
            now = time.monotonic()
            if steps % (5 if DRY else 200) == 0:
                print("STEP", json.dumps({"cand": config["name"], "steps": steps, "sec": round(now - start), "lr": opt.param_groups[0]["lr"], "sec_per_step": round((now - start) / steps, 3),
                                          **{k: round(v / steps, 4) for k, v in run.items()}}), flush=True)
            final = now - start > TRAIN_SECONDS
            if now - last_eval > EVAL_EVERY or final:
                metrics = evaluate(student, pages, device)
                value = score(metrics)
                row = {"config": config, "steps": steps, "seconds": round(now - start), "score": value, "validation": metrics}
                log.append(row)
                print("VALID", json.dumps({"cand": config["name"], "steps": steps, "score": round(value, 5), **{k: round(v, 3) for k, v in metrics.items()}}), flush=True)
                if value > best:
                    best = value
                    torch.save({"state_dict": student.state_dict(), "selection": row}, OUT / f"ckpt-{config['name']}.pt")
                (OUT / f"history-{config['name']}.json").write_text(json.dumps(log, indent=1))
                last_eval = time.monotonic()
            if final:
                done = True
                break
    queue.put({"name": config["name"], "best": best, "steps": steps, "sec_per_step": (time.monotonic() - start) / max(steps, 1)})


if __name__ == "__main__":
    # Query GPUs without initialising CUDA here: forked candidates cannot use CUDA if the parent did.
    gpus = [] if DRY else [l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU ")]
    print("DEVICES", len(gpus), gpus, flush=True)
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    configs = CONFIGS[:1] if DRY else CONFIGS
    for wave in range(0, len(configs), max(1, len(gpus))):
        procs = []
        for rank, cfg in enumerate(configs[wave:wave + max(1, len(gpus))]):
            p = ctx.Process(target=candidate, args=(rank, cfg, queue))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            assert p.exitcode == 0, f"candidate process failed ({p.exitcode})"
    summaries = []
    while not queue.empty():
        summaries.append(queue.get())
    print("CANDIDATES", json.dumps(summaries), flush=True)
    parity = verify_parity(ORIGINAL)  # after the forks: keeps OpenMP out of the parent before fork
    print("PARITY", json.dumps(parity), flush=True)

    # ---------------- selection on validation only (the original is eligible)
    base = json.loads((OUT / "baseline.json").read_text())
    options = [("start", score(base), None)]
    results = {"baseline": base, "start_parity": parity, "candidates": summaries, "configs": configs, "model_revision": MODEL_REV, "levels": EVAL_LEVELS, "ious": IOUS,
               "pages": {s: dict(collections.Counter(i["source"] for i in v)) for s, v in ITEMS.items()}}
    for cfg in configs:
        path = OUT / f"ckpt-{cfg['name']}.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        options.append((cfg["name"], ckpt["selection"]["score"], path))
        results[f"selection_{cfg['name']}"] = ckpt["selection"]
        model = DetNet(ORIGINAL)
        model.load_state_dict(ckpt["state_dict"])
        model.write_onnx(ORIGINAL, OUT / f"det-{cfg['name']}.onnx")
        results[f"export_parity_{cfg['name']}"] = verify_parity(OUT / f"det-{cfg['name']}.onnx", model)
    options.sort(key=lambda o: -o[1])
    results["selected"] = options[0][0]
    results["selection_scores"] = {n: s for n, s, _ in options}
    print("SELECTION", json.dumps([(n, round(s, 5)) for n, s, _ in options]), "->", options[0][0], flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf8")
    print("FINAL", json.dumps({"selected": options[0][0]}), flush=True)
