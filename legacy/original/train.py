"""Fine-tune only Indra's RTL line recognizer; retain its PDF pipeline."""
import collections
import concurrent.futures
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import unicodedata

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq
from PIL import Image
from rapidfuzz.distance import Levenshtein
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from huggingface_hub import hf_hub_download

from indra_rtl_trainable import IndraRTL, verify_parity
from indra_ocr.rec_preproc import crop_line, pad_batch
from indra_ocr.rtl import flip, normalize, visual_label

SEED = 20261101
MODEL_REV = "bd02167fdf2005a19962cc15188f8496a2c87d3b"
CODE_REV = "e20355319b5aa5705a876a8d2b40aca606d7df5f"
OUT = Path(os.environ.get("OCR_OUTPUT", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)
TRAIN_SECONDS = int(os.environ.get("OCR_TRAIN_SECONDS", "5400"))
EPOCHS = int(os.environ.get("OCR_EPOCHS", "8"))
BATCH = int(os.environ.get("OCR_BATCH_SIZE", "64"))
PEAK_LR = float(os.environ.get("OCR_MAX_LR", "0.0002"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)
cv2.setNumThreads(1)

def clean(text):
    text = unicodedata.normalize("NFC", normalize(text))
    text = "".join(c for c in text if unicodedata.category(c) != "Cf" or c == "\u200c")
    return " ".join(text.split())

def canonical(text):
    return text.translate(str.maketrans({"ي":"ی", "ك":"ک", "\u200c":""})).replace(" ", "")

def preprocess(row):
    text = clean(row["text"])
    gray = np.asarray(Image.open(io.BytesIO(row["image"])).convert("L"))
    # Match Indra's detector ink box and shared line-padding rule.
    if np.median(gray) < 110:
        gray = 255 - gray
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, dtype=np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 2).astype(np.uint8)
    ys, xs = np.where(keep[labels] > 0)
    if len(xs) < 3 or not text:
        return None, "empty"
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    ink = crop_line(gray, box, max_width=1536, dark=False)
    label = visual_label(text)
    required = len(label) + sum(a == b for a,b in zip(label, label[1:]))
    if required > math.ceil(ink.shape[1] / 4):
        return None, "ctc_too_short"
    return {"ink": np.rint(ink * 255).astype(np.uint8), "text":text, "visual":label, "source":row["source"], "source_id":row["source_id"]}, None

def load_rows(path):
    rows = pq.read_table(path).to_pylist()
    rejected = collections.Counter()
    result = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for item, reason in pool.map(preprocess, rows, chunksize=64):
            if item is None:
                rejected[reason] += 1
            else:
                result.append(item)
    print("Loaded", path.name, len(result), "rejected", dict(rejected), flush=True)
    return result, {"input":len(rows), "retained":len(result), "rejected":dict(rejected)}

class Lines(Dataset):
    def __init__(self, rows, chars, augment=False):
        self.rows = rows
        self.ids = {c:i + 1 for i,c in enumerate(chars)}
        self.augment = augment
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]
        ink = row["ink"]
        if self.augment and random.random() < 0.35:
            ink = ink.copy()
            if random.random() < 0.5:
                ink = cv2.GaussianBlur(ink, (3,3), random.uniform(0.25,0.8))
            if random.random() < 0.4:
                scale = random.uniform(0.65,0.95)
                small = cv2.resize(ink, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                ink = cv2.resize(small, (ink.shape[1], 32), interpolation=cv2.INTER_LINEAR)
        return ink, [self.ids.get(c, 0) for c in row["visual"]], row

def collate(items):
    inks, labels, rows = zip(*items)
    x, widths = pad_batch([ink.astype(np.float32) / 255 for ink in inks])
    return torch.from_numpy(x), torch.from_numpy(widths), torch.tensor([x for seq in labels for x in seq], dtype=torch.long), torch.tensor([len(seq) for seq in labels], dtype=torch.long), rows

class WidthBatches(Sampler):
    def __init__(self, rows, shuffle):
        self.order = sorted(range(len(rows)), key=lambda i: rows[i]["ink"].shape[1])
        self.shuffle = shuffle
    def __iter__(self):
        groups = [self.order[i:i + BATCH] for i in range(0,len(self.order),BATCH)]
        if self.shuffle:
            random.shuffle(groups)
            for group in groups:
                random.shuffle(group)
        yield from groups
    def __len__(self):
        return math.ceil(len(self.order)/BATCH)

def make_loader(rows, chars, train=False):
    return DataLoader(Lines(rows, chars, train), batch_sampler=WidthBatches(rows, train), collate_fn=collate, num_workers=2, pin_memory=True, persistent_workers=True)

def book_long_lines(rows):
    """Join neighboring RTL book lines into wider supervised training examples."""
    pages = collections.defaultdict(list)
    for row in rows:
        if row["source"].startswith("arshasb/"):
            page,line = row["source_id"].split(":")
            pages[page].append((int(line),row))
    result = []
    for page,lines in pages.items():
        lines.sort(key=lambda x:x[0])
        for (ai,a),(bi,b) in zip(lines[::2],lines[1::2]):
            if bi != ai+1:
                continue
            def strip_x(ink):
                columns = np.flatnonzero(ink.max(0)>40)
                return ink[:,max(0,columns[0]-2):columns[-1]+3] if len(columns) else ink
            # Logical first line is on the right; both retain the same baseline.
            ink = np.concatenate([strip_x(b["ink"]),np.zeros((32,6),np.uint8),strip_x(a["ink"])],axis=1)
            text = a["text"]+" "+b["text"]
            visual = visual_label(text)
            required = len(visual)+sum(x==y for x,y in zip(visual,visual[1:]))
            if ink.shape[1] <= 2048 and required <= math.ceil(ink.shape[1]/4):
                result.append({"ink":ink,"text":text,"visual":visual,"source":"arshasb/Far_ketab_long","source_id":f"{page}:{ai}+{bi}"})
    return result

def selection_score(metrics):
    if os.environ.get("OCR_CHECKPOINT_METRIC") != "balanced_book":
        return metrics["all"]["cer"]
    book = [v for k,v in metrics.items() if k.startswith("arshasb/")]
    other = [v for k,v in metrics.items() if k != "all" and not k.startswith("arshasb/")]
    return 0.5*sum(x["char_edits"] for x in book)/sum(x["chars"] for x in book)+0.5*sum(x["char_edits"] for x in other)/sum(x["chars"] for x in other)

def decode(logits, lengths, chars):
    indices = logits.argmax(-1).cpu().numpy()
    result = []
    for seq, size in zip(indices, lengths.cpu().tolist()):
        previous = -1
        text = []
        for i in seq[:size]:
            if i and i != previous:
                text.append(chars[i - 1])
            previous = i
        result.append(clean(flip("".join(text))))
    return result

@torch.inference_mode()
def evaluate(model, loader, chars, predictions_path=None):
    model.eval()
    totals = collections.defaultdict(collections.Counter)
    predictions = []
    for x, widths, labels, label_lengths, rows in loader:
        logits, lengths = model(x.cuda(non_blocking=True), widths.cuda(non_blocking=True))
        texts = decode(logits, lengths, chars)
        for row, prediction in zip(rows, texts):
            reference = row["text"]
            for key in ("all", row["source"]):
                t = totals[key]
                t["char_edits"] += Levenshtein.distance(reference,prediction)
                t["chars"] += len(reference)
                t["word_edits"] += Levenshtein.distance(reference.split(),prediction.split())
                t["words"] += len(reference.split())
                t["canonical_edits"] += Levenshtein.distance(canonical(reference),canonical(prediction))
                t["canonical_chars"] += len(canonical(reference))
                t["exact"] += reference == prediction
                t["samples"] += 1
            if predictions_path:
                predictions.append({"source":row["source"], "source_id":row["source_id"], "reference":reference, "prediction":prediction})
    result = {k:{**dict(t), "cer":t["char_edits"]/max(1,t["chars"]), "wer":t["word_edits"]/max(1,t["words"]), "canonical_no_space_cer":t["canonical_edits"]/max(1,t["canonical_chars"]), "exact_accuracy":t["exact"]/max(1,t["samples"])} for k,t in totals.items()}
    if predictions_path:
        predictions_path.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in predictions), encoding="utf-8")
    return result

def main():
    assert torch.cuda.is_available(), "GPU unavailable: refusing CPU training"
    device = {"torch":torch.__version__, "cuda":torch.version.cuda, "gpu_count":torch.cuda.device_count(), "gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    print("GPU", json.dumps(device), flush=True)
    model_dir = OUT / "model"
    model_dir.mkdir(exist_ok=True)
    base_dir = Path("/tmp/indra-base")
    base_dir.mkdir(exist_ok=True)
    filenames = ["det.onnx","rec.onnx","rec_cjk.onnx","rec_world.onnx","rec_rtl.onnx","order.onnx","orient.onnx","script.onnx","meta.json","lm.npz","lm_cjk.npz","lm_world.npz","LICENSE"]
    def fetch(name):
        p = hf_hub_download("datalab-to/indra-ocr",name,revision=MODEL_REV)
        shutil.copy2(p, base_dir / name)
        shutil.copy2(p, model_dir / name)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        list(pool.map(fetch, filenames))
    original = base_dir / "rec_rtl.onnx"
    parity = verify_parity(original)
    print("Pretrained parity", json.dumps(parity), flush=True)
    candidates = list(Path("/kaggle/input").rglob("train.parquet"))
    assert len(candidates) == 1, f"Expected exactly one corpus, found {candidates}"
    corpus = candidates[0].parent
    for name in ("data-report.json","ATTRIBUTION.md"):
        shutil.copy2(corpus/name, OUT/name)
    splits, coverage = {}, {}
    for split in ("train","validation","test"):
        splits[split], coverage[split] = load_rows(corpus / f"{split}.parquet")
    if os.environ.get("OCR_BOOK_LONG_LINES") == "1":
        extra_lines = book_long_lines(splits["train"])
        splits["train"].extend(extra_lines)
        coverage["train"]["long_line_augmentations"] = len(extra_lines)
        print("Added book long lines",len(extra_lines),flush=True)
    # Verify the uploaded corpus split independently, before touching its labels.
    hashes = {split:set(pq.read_table(corpus/f"{split}.parquet",columns=["text_group"])["text_group"].to_pylist()) for split in splits}
    assert not (hashes["train"] & hashes["validation"] or hashes["train"] & hashes["test"] or hashes["validation"] & hashes["test"]), "Text leakage across splits"
    meta = json.loads((base_dir/"meta.json").read_text())
    chars = json.loads(meta["vocab_rtl"])["chars"]
    original_chars = list(chars)
    extra = sorted({c for values in splits.values() for row in values for c in row["visual"]} - set(chars))
    # Vocabulary discovery uses only training labels; rows with unseen eval-only
    # characters are still scored, but cannot invent new trainable classes.
    train_chars = {c for row in splits["train"] for c in row["visual"]}
    extra = [c for c in extra if c in train_chars or c == "\u200c"]
    if "\u200c" not in chars and "\u200c" not in extra:
        extra.append("\u200c")
    model = IndraRTL.from_onnx(original)
    baseline_loaders = {split:make_loader(splits[split], original_chars) for split in ("validation","test")}
    # Evaluation does not use numeric labels, so tolerate characters the base
    # model cannot emit while keeping the full reference for CER/WER.
    model.cuda()
    baseline = {split:evaluate(model,baseline_loaders[split],original_chars) for split in baseline_loaders}
    (OUT/"baseline-metrics.json").write_text(json.dumps(baseline,indent=2))
    print("BASELINE", json.dumps({k:v["all"] for k,v in baseline.items()}), flush=True)
    model.add_characters(extra, original_chars.index(" ")+1)
    if os.environ.get("OCR_FREEZE_BACKBONE") == "1":
        for name,parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(("blocks.8.","blocks.9.","head.")))
        print("Trainable parameters",sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
    chars += extra
    meta["vocab_rtl"] = json.dumps({"blank":0,"chars":chars},ensure_ascii=False)
    (model_dir/"meta.json").write_text(json.dumps(meta,ensure_ascii=False), encoding="utf-8")
    loaders = {split:make_loader(splits[split],chars,split=="train") for split in splits}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=PEAK_LR/2,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=PEAK_LR,total_steps=EPOCHS*len(loaders["train"]),pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    best = float("inf")
    history = []
    started = time.monotonic()
    stale = 0
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        batches = 0
        for x,widths,labels,label_lengths,_ in loaders["train"]:
            if time.monotonic()-started > TRAIN_SECONDS:
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                logits,lengths = model(x.cuda(non_blocking=True),widths.cuda(non_blocking=True))
            loss = F.ctc_loss(logits.float().log_softmax(-1).transpose(0,1),labels.cuda(non_blocking=True),lengths.cpu(),label_lengths,blank=0,zero_infinity=False)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite CTC loss at epoch {epoch+1}, batch {batches}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            batches += 1
            if batches % 200 == 0:
                print(json.dumps({"epoch":epoch+1,"batch":batches,"loss":running/batches,"elapsed_seconds":round(time.monotonic()-started)}),flush=True)
        validation = evaluate(model,loaders["validation"],chars)
        record = {"epoch":epoch+1,"loss":running/max(1,batches),"batches":batches,"seconds":round(time.monotonic()-started),"validation":validation}
        history.append(record)
        print("EPOCH",json.dumps(record),flush=True)
        if selection_score(validation) < best:
            best = selection_score(validation)
            stale = 0
            torch.save({"state_dict":model.state_dict(),"chars":chars,"epoch":epoch+1,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"model_revision":MODEL_REV},OUT/"best-checkpoint.pt")
            model.write_onnx(original, model_dir/"rec_rtl.onnx")
        else:
            stale += 1
        (OUT/"training-history.json").write_text(json.dumps(history,indent=2))
        if stale >= 3 or time.monotonic()-started > TRAIN_SECONDS:
            break
    checkpoint = torch.load(OUT/"best-checkpoint.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    final = {split:evaluate(model,loaders[split],chars,OUT/f"{split}-predictions.jsonl") for split in ("validation","test")}
    model.cpu().eval()
    exported_parity = verify_parity(model_dir/"rec_rtl.onnx",model,atol=5e-4)
    report = {"device":device,"model_revision":MODEL_REV,"code_revision":CODE_REV,"pretrained_parity":parity,"exported_parity":exported_parity,"coverage":coverage,"added_characters":extra,"selected_epoch":checkpoint["epoch"],"training_seconds":round(time.monotonic()-started),"baseline":baseline,"fine_tuned":final,"better_validation":final["validation"]["all"]["cer"]<baseline["validation"]["all"]["cer"],"better_test":final["test"]["all"]["cer"]<baseline["test"]["all"]["cer"],"configuration":{k:v for k,v in os.environ.items() if k.startswith("OCR_")},"scope":"Synthetic printed Persian line recognition; real scanned books and full-page layout require separate evaluation."}
    (OUT/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    shutil.make_archive(str(OUT/"indra-persian-model"),"zip",model_dir)
    print("FINAL",json.dumps(report,ensure_ascii=False),flush=True)

if __name__ == "__main__":
    main()
