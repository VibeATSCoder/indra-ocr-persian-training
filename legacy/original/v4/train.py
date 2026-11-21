"""v4: regularised Indra RTL fine-tuning, one candidate per T4, validation-only selection.

Data: persian-ocr-large-mixed-v3 (748K lines) + v4 book-font render (scan-degraded lines in
13 free Persian fonts; Nahid = validation family, Nazli = test family).
Objective (per GPU, no DataParallel): CTC + KL to the frozen original model + L2 anchor.
Selection: weighted validation error over real / font / bookfont / synthetic domains.
The original model is an eligible candidate. Test splits are opened only after selection.

DRY_RUN=1 runs the whole pipeline on CPU with a tiny subset (local check, no Kaggle).
"""
import collections
import concurrent.futures
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import requests
import torch
from rapidfuzz.distance import Levenshtein
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from common import canonical, clean
from indra_ocr.rec_preproc import pad_batch
from indra_ocr.rtl import flip
from indra_rtl_trainable import IndraRTL, verify_parity

DRY = os.environ.get("DRY_RUN") == "1"
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("KAGGLE_OUT", "/kaggle/working"))
TMP = Path(os.environ.get("TRAIN_TMP", "/tmp/v4-train"))
TMP.mkdir(parents=True, exist_ok=True)
SEED = 20261104
BATCH = 16 if DRY else 192
EVAL_EVERY = 30 if DRY else 1100  # seconds between validation passes
ORIGINAL_CLASSES = 1356
MODEL_REV = "bd02167fdf2005a19962cc15188f8496a2c87d3b"
MASS = {"real": .15, "font": .12, "bookfont": .25, "arshasb": .12, "persian230k": .14, "garshasp": .22}
WEIGHTS = {"real": .35, "bookfont": .30, "font": .25, "garshasp": .04, "arshasb": .03, "persian230k": .03}
RUN = os.environ.get("RUN", "v4")
# Candidate grid per Kaggle run; candidates train two at a time (one per T4).
# v5 crosses learning rate x regularisation: v4 validation stopped improving within ~20 min at both of its settings.
RUNS = {"v4": (4200, [{"name": "adapt", "lr": 3e-5, "anchor": 1e-3, "distill": .15},
                      {"name": "conservative", "lr": 1e-5, "anchor": 3e-3, "distill": .30}]),
        "v5": (4500, [{"name": "adapt", "lr": 3e-5, "anchor": 1e-3, "distill": .15},
                      {"name": "loose", "lr": 3e-5, "anchor": 2e-4, "distill": .05},
                      {"name": "fast", "lr": 8e-5, "anchor": 1e-3, "distill": .15},
                      {"name": "fast-loose", "lr": 8e-5, "anchor": 2e-4, "distill": .05}])}
TRAIN_SECONDS = float(os.environ.get("TRAIN_SECONDS", 60 if DRY else RUNS[RUN][0]))  # per candidate
CONFIGS = RUNS[RUN][1]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)
cv2.setNumThreads(1)

# ------------------------------------------------------------------ base model
base = TMP / "base"
base.mkdir(exist_ok=True)
FILES = ["det.onnx", "rec.onnx", "rec_cjk.onnx", "rec_world.onnx", "rec_rtl.onnx", "order.onnx", "orient.onnx",
         "script.onnx", "meta.json", "lm.npz", "lm_cjk.npz", "lm_world.npz", "LICENSE"]


def download(name):
    path = base / name
    if not path.exists():
        r = requests.get(f"https://huggingface.co/datalab-to/indra-ocr/resolve/{MODEL_REV}/{name}", timeout=180)
        r.raise_for_status()
        path.write_bytes(r.content)


with concurrent.futures.ThreadPoolExecutor(4) as pool:
    list(pool.map(download, FILES if not DRY else ["rec_rtl.onnx", "meta.json"]))
ORIGINAL = base / "rec_rtl.onnx"
ORIGINAL_CHARS = json.loads(json.loads((base / "meta.json").read_text())["vocab_rtl"])["chars"]

# ------------------------------------------------------------------ data cache (CPU, parallel per shard)
v3_manifest_path = next(INPUT.rglob("manifest.json"))
v3 = json.loads(v3_manifest_path.read_text())
CHARS = v3["characters"]
TOKEN = {c: i + 1 for i, c in enumerate(CHARS)}
shards = [(v3_manifest_path.parent / s["file"], s["split"]) for s in v3["shards"]]
book_manifest_path = next(INPUT.rglob("bookfont-manifest.json"), None)
book = json.loads(book_manifest_path.read_text()) if book_manifest_path else {"shards": []}
shards += [(book_manifest_path.parent / s["file"], s["split"]) for s in book["shards"]]
if DRY:
    shards = shards[: int(os.environ.get("DRY_SHARDS", 99))]
print("SHARDS", len(shards), "bookfont", len(book["shards"]), flush=True)


def cache_shard(job):
    index, (path, split) = job
    bin_path = TMP / f"ink-{index:03d}.bin"
    rows, offset = [], 0
    with bin_path.open("wb", buffering=8 << 20) as handle:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                data = row.pop("ink")
                if len(data) != 32 * row["width"]:
                    continue
                if random.random() < .01:  # spot-check integrity of the published cache
                    assert hashlib.sha256(data).hexdigest() == row["image_sha256"], "cache integrity"
                handle.write(data)
                rows.append((split, index, offset, row["width"], row["text"], row["visual"], row["domain"],
                             row["doc_id"], row["group"], row["image_sha256"], row["source"]))
                offset += len(data)
                if DRY and len(rows) >= 300:
                    return rows
    return rows


t0 = time.monotonic()
with mp.get_context("fork").Pool(min(4, os.cpu_count() or 4)) as pool:
    cached = pool.map(cache_shard, list(enumerate(shards)))
print("CACHED", sum(map(len, cached)), "rows in", round(time.monotonic() - t0), "s", flush=True)

# Resolve label conflicts (test > validation > train) and enforce disjointness.
groups = collections.defaultdict(set)
for rows in cached:
    for r in rows:
        groups[r[0]].add(r[8])
records, conflicts = collections.defaultdict(list), collections.Counter()
images, fonts, docs = collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set)
for rows in cached:
    for split, idx, off, width, text, visual, domain, doc, group, sha, source in rows:
        if split == "train" and (group in groups["validation"] or group in groups["test"]):
            conflicts["train_label_in_eval"] += 1
            continue
        if split == "validation" and group in groups["test"]:
            conflicts["validation_label_in_test"] += 1
            continue
        if sha in images["any"]:
            conflicts["duplicate_image"] += 1
            continue
        images["any"].add(sha)
        try:
            target = [TOKEN[c] for c in visual]
        except KeyError:
            conflicts["unknown_char"] += 1
            continue
        records[split].append({"file": idx, "offset": off, "width": width, "text": text, "target": target,
                               "domain": domain, "doc_id": doc, "group": group, "source": source})
        if domain in ("font", "bookfont"):
            fonts[split].add(doc)
        if domain == "real":
            docs[split].add(doc)
del cached, groups, images
for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert not fonts[a] & fonts[b], f"font family overlap {a}/{b}: {fonts[a] & fonts[b]}"
    assert not docs[a] & docs[b], f"real document overlap {a}/{b}"
COUNTS = {s: dict(collections.Counter(r["domain"] for r in v)) for s, v in records.items()}
print("SPLITS", json.dumps(COUNTS), "conflicts", json.dumps(conflicts), flush=True)
MAPS = {}


def ink(row):
    m = MAPS.get(row["file"])
    if m is None:
        m = MAPS[row["file"]] = np.memmap(TMP / f"ink-{row['file']:03d}.bin", dtype=np.uint8, mode="r")
    return np.asarray(m[row["offset"]: row["offset"] + 32 * row["width"]]).reshape(32, row["width"])


def augment(im, r):
    """Scan-like perturbations on the 32px crop (ink bright on dark, as Indra crops)."""
    im = im.copy()
    if r.random() < .4:
        s = r.uniform(.5, 1)
        small = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        im = cv2.resize(small, (im.shape[1], 32), interpolation=cv2.INTER_LINEAR)
    if r.random() < .35:
        im = cv2.GaussianBlur(im, (3, 3), r.uniform(.2, .9))
    if r.random() < .15:
        im = (cv2.erode if r.random() < .5 else cv2.dilate)(im, np.ones((2, 1), np.uint8))
    if r.random() < .35:
        im = np.clip(im.astype(np.float32) * r.uniform(.6, 1.15) + r.uniform(-10, 15)
                     + np.random.normal(0, r.uniform(0, 6), im.shape), 0, 255).astype(np.uint8)
    if r.random() < .2:
        ok, buf = cv2.imencode(".jpg", 255 - im, [cv2.IMWRITE_JPEG_QUALITY, r.randint(30, 90)])
        im = 255 - cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if r.random() < .25:
        mat = np.float32([[1, r.uniform(-.03, .03), 0], [0, 1, r.uniform(-1, 1)]])
        im = cv2.warpAffine(im, mat, (im.shape[1], 32), borderValue=0)
    return im


class Lines(Dataset):
    def __init__(self, rows, train=False):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        im = ink(row)
        if self.train and random.random() < .7:
            im = augment(im, random)
        return im, row


def collate(items):
    x, widths = pad_batch([im.astype(np.float32) / 255 for im, _ in items])
    rows = [r for _, r in items]
    labels = torch.tensor([t for r in rows for t in r["target"]], dtype=torch.long)
    sizes = torch.tensor([len(r["target"]) for r in rows], dtype=torch.long)
    return torch.from_numpy(x), torch.from_numpy(widths), labels, sizes, rows


class Plan(Sampler):
    """Domain-balanced sampling with replacement, width-bucketed batches."""

    def __init__(self, rows, train, seed):
        self.rows, self.train, self.seed, self.epoch = rows, train, seed, 0
        self.n = 2_000_000 if train else len(rows)
        if train:
            count = collections.Counter(r["domain"] for r in rows)
            mass = {d: MASS.get(d, .05) for d in count}
            p = np.asarray([mass[r["domain"]] / count[r["domain"]] for r in rows])
            self.p = p / p.sum()

    def __len__(self):
        return math.ceil(self.n / BATCH)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.choice(len(self.rows), self.n, replace=True, p=self.p) if self.train else np.arange(self.n)
        batches = []
        for start in range(0, self.n, 8192):
            pool = sorted(idx[start:start + 8192].tolist(), key=lambda i: self.rows[i]["width"])
            batches += [pool[i:i + BATCH] for i in range(0, len(pool), BATCH)]
        if self.train:
            rng.shuffle(batches)
        yield from batches


def loader(split, train=False, seed=SEED, workers=2):
    return DataLoader(Lines(records[split], train), batch_sampler=Plan(records[split], train, seed), collate_fn=collate,
                      num_workers=0 if DRY else workers, pin_memory=not DRY, persistent_workers=not DRY and train)


def decode(logits, lengths, chars):
    out = []
    for seq, n in zip(logits.argmax(-1).cpu().numpy(), lengths.cpu().tolist()):
        last, s = -1, []
        for t in seq[:n]:
            if t and t != last:
                s.append(chars[int(t) - 1])
            last = t
        out.append(clean(flip("".join(s))))
    return out


@torch.inference_mode()
def evaluate(model, data, chars, device, save=None):
    model.eval()
    sums, preds = collections.defaultdict(collections.Counter), []
    for x, widths, _, _, rows in data:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, lengths = model(x.to(device), widths.to(device))
        for row, p in zip(rows, decode(logits.float(), lengths, chars)):
            t = row["text"]
            m = {"char_edits": Levenshtein.distance(t, p), "chars": len(t), "word_edits": Levenshtein.distance(t.split(), p.split()),
                 "words": len(t.split()), "canonical_edits": Levenshtein.distance(canonical(t), canonical(p)),
                 "canonical_chars": len(canonical(t)), "samples": 1, "exact": int(t == p)}
            for d in ("all", row["domain"], row["source"] if row["domain"] in ("bookfont", "font") else None):
                if d:
                    sums[d].update(m)
            if save is not None:
                preds.append({"source": row["source"], "reference": t, "prediction": p})
    res = {k: {**dict(v), "cer": v["char_edits"] / max(1, v["chars"]), "wer": v["word_edits"] / max(1, v["words"]),
               "canonical_cer": v["canonical_edits"] / max(1, v["canonical_chars"])} for k, v in sums.items()}
    if save is not None:
        save.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in preds), encoding="utf8")
    return res


def score(metrics):
    valid = {k: w for k, w in WEIGHTS.items() if k in metrics and metrics[k]["samples"] >= (5 if DRY else 100)}
    return sum(w * (.7 * metrics[k]["cer"] + .3 * metrics[k]["canonical_cer"]) for k, w in valid.items()) / sum(valid.values())


def new_student():
    m = IndraRTL.from_onnx(ORIGINAL)
    m.add_characters(CHARS[len(ORIGINAL_CHARS):], ORIGINAL_CHARS.index(" ") + 1)
    return m


def candidate(rank, config, queue):
    """One process per GPU. Trains `config`, validates periodically, keeps its best checkpoint."""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and not DRY else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed = SEED + 1000 * rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    teacher = IndraRTL.from_onnx(ORIGINAL).to(device).eval().requires_grad_(False)
    val = loader("validation")
    log = []
    if rank == 0:  # baseline for selection, computed once
        base_val = evaluate(teacher, val, ORIGINAL_CHARS, device)
        (OUT / "baseline-validation.json").write_text(json.dumps(base_val, indent=1))
        print("BASELINE", json.dumps({k: round(v["cer"], 4) for k, v in base_val.items()}), flush=True)
    student = new_student().to(device)
    anchors = {n: p.detach().clone() for n, p in student.named_parameters()}
    opt = torch.optim.AdamW(student.parameters(), lr=config["lr"], weight_decay=0)

    def lr_at(frac):  # time-based: 3% warmup, cosine to lr/50 exactly at TRAIN_SECONDS (v4 run 1 decayed too early)
        if frac < .03:
            return config["lr"] * (.1 + .9 * frac / .03)
        return config["lr"] * (.02 + .98 * .5 * (1 + math.cos(math.pi * min(1, (frac - .03) / .97))))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    train = loader("train", True, seed, workers=2)
    start = last_eval = time.monotonic()
    best, steps, seen, run = float("inf"), 0, 0, collections.Counter()
    done = False
    while not done:
        for x, widths, labels, sizes, rows in train:
            student.train()
            x, widths = x.to(device, non_blocking=True), widths.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, lengths = student(x, widths)
                with torch.no_grad():
                    old, _ = teacher(x, widths)
            lp = logits.float().log_softmax(-1)
            ctc = F.ctc_loss(lp.transpose(0, 1), labels.to(device), lengths, sizes.to(device), blank=0, zero_infinity=True)
            mask = (torch.arange(logits.shape[1], device=device)[None, :] < lengths[:, None]).float()
            kd = (F.kl_div(F.log_softmax(logits[:, :, :ORIGINAL_CLASSES].float() / 2, -1), F.softmax(old.float() / 2, -1),
                           reduction="none").sum(-1) * mask).sum() / mask.sum().clamp_min(1) * 4
            anchor = sum((p - anchors[n]).square().sum() for n, p in student.named_parameters())
            loss = ctc + config["distill"] * kd + config["anchor"] * anchor
            if not torch.isfinite(loss):
                run["nonfinite_skipped"] += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
            for g in opt.param_groups:
                g["lr"] = lr_at((time.monotonic() - start) / TRAIN_SECONDS)
            scaler.step(opt)
            scaler.update()
            steps += 1
            seen += len(rows)
            run.update({"loss": loss.item(), "ctc": ctc.item(), "kl": kd.item(), "anchor": anchor.item()})
            now = time.monotonic()
            if steps % (5 if DRY else 400) == 0:
                print("STEP", json.dumps({"cand": config["name"], "steps": steps, "seen": seen, "sec": round(now - start), "lr": opt.param_groups[0]["lr"],
                                          "sec_per_step": round((now - start) / steps, 3),
                                          **{k: round(v / steps, 4) for k, v in run.items()}}), flush=True)
            final = now - start > TRAIN_SECONDS
            if now - last_eval > EVAL_EVERY or final:
                metrics = evaluate(student, val, CHARS, device)
                value = score(metrics)
                row = {"config": config, "steps": steps, "seen": seen, "seconds": round(now - start), "score": value,
                       "validation": {k: {kk: metrics[k][kk] for kk in ("cer", "canonical_cer", "samples")} for k in metrics}}
                log.append(row)
                print("VALID", json.dumps({"cand": config["name"], "steps": steps, "score": round(value, 5),
                                           **{k: round(metrics[k]["cer"], 4) for k in metrics if k in WEIGHTS or k == "all"}}), flush=True)
                if value < best:
                    best = value
                    torch.save({"state_dict": student.state_dict(), "characters": CHARS, "selection": row},
                               OUT / f"ckpt-{config['name']}.pt")
                (OUT / f"history-{config['name']}.json").write_text(json.dumps(log, indent=1))
                last_eval = time.monotonic()
            if final:
                done = True
                break
    queue.put({"name": config["name"], "best": best, "steps": steps, "seen": seen,
               "sec_per_step": (time.monotonic() - start) / max(steps, 1)})


if __name__ == "__main__":
    # Query GPUs without initialising CUDA here: forked candidates cannot use CUDA if the parent did.
    import subprocess
    gpus = [] if DRY else [l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU ")]
    ngpu = len(gpus)
    print("DEVICES", ngpu, gpus, flush=True)
    ctx = mp.get_context("fork")  # CUDA is untouched in the parent until candidates finish
    queue = ctx.Queue()
    procs = []
    slots = max(1, ngpu)
    summaries = []
    for wave in range(0, len(CONFIGS), slots):
        for rank, cfg in enumerate(CONFIGS[wave:wave + slots]):
            p = ctx.Process(target=candidate, args=(rank, cfg, queue))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            assert p.exitcode == 0, f"candidate process failed ({p.exitcode})"
        procs = []
    while not queue.empty():
        summaries.append(queue.get())
    print("CANDIDATES", json.dumps(summaries), flush=True)
    parity = verify_parity(ORIGINAL)  # after the forks: keeps OpenMP out of the parent before fork
    print("PARITY", json.dumps(parity), flush=True)

    # ---------------- selection on validation only (original is eligible)
    device = torch.device("cuda:0" if ngpu else "cpu")
    base_val = json.loads((OUT / "baseline-validation.json").read_text())
    options = [("original", score(base_val), None)]
    for cfg in CONFIGS:
        path = OUT / f"ckpt-{cfg['name']}.pt"
        if path.exists():
            options.append((cfg["name"], torch.load(path, map_location="cpu", weights_only=False)["selection"]["score"], path))
    options.sort(key=lambda o: o[1])
    selected = options[0]
    print("SELECTION", json.dumps([(n, round(s, 5)) for n, s, _ in options]), "->", selected[0], flush=True)

    # ---------------- open test only now
    test = loader("test")
    teacher = IndraRTL.from_onnx(ORIGINAL).to(device).eval()
    results = {"baseline_test": evaluate(teacher, test, ORIGINAL_CHARS, device, OUT / "baseline-test-predictions.jsonl")}
    bundles = {}
    for name, value, path in options:
        if path is None:
            continue
        model = new_student()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
        model.to(device).eval()
        results[f"test_{name}"] = evaluate(model, test, CHARS, device, OUT / f"test-predictions-{name}.jsonl")
        bundle = OUT / f"model-{name}"
        bundle.mkdir(exist_ok=True)
        for f in FILES:
            if (base / f).exists():
                shutil.copyfile(base / f, bundle / f)
        model.cpu().eval()
        model.write_onnx(ORIGINAL, bundle / "rec_rtl.onnx")
        meta = json.loads((base / "meta.json").read_text())
        meta["vocab_rtl"] = json.dumps({"blank": 0, "chars": CHARS}, ensure_ascii=False)
        (bundle / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf8")
        results[f"export_parity_{name}"] = verify_parity(bundle / "rec_rtl.onnx", model, atol=5e-4)
        shutil.make_archive(str(OUT / f"indra-persian-{RUN}-{name}"), "zip", bundle)
        bundles[name] = str(bundle)
    report = {"selected": selected[0], "selection_scores": {n: s for n, s, _ in options}, "candidates": summaries,
              "configs": CONFIGS, "mass": MASS, "weights": WEIGHTS, "counts": COUNTS, "conflicts_removed": dict(conflicts),
              "font_families": {k: sorted(v) for k, v in fonts.items()}, "train_seconds_per_candidate": TRAIN_SECONDS,
              "pretrained_parity": parity, "run": RUN, "model_revision": MODEL_REV, "bundles": bundles,
              "baseline_validation": base_val, **results,
              "notes": "Test opened only after validation selection. Real-domain labels are machine-generated (filtered), not human ground truth."}
    (OUT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf8")
    short = {k: (round(v["all"]["cer"], 4) if isinstance(v, dict) and "all" in v else v) for k, v in results.items() if k.startswith(("baseline", "test"))}
    print("FINAL", json.dumps({"selected": selected[0], "test_all_cer": short}), flush=True)
