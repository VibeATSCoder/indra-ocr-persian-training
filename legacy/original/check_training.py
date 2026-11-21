"""Exercise actual corpus preprocessing, RTL labels, gradients and ONNX export."""
import json
from pathlib import Path
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
import train
from indra_rtl_trainable import IndraRTL, verify_parity

torch.set_num_threads(2)
rows = pq.ParquetFile("corpus/test.parquet").read_row_group(0).to_pylist()[:16]
processed = [train.preprocess(row)[0] for row in rows]
processed = [row for row in processed if row is not None]
meta = json.loads(Path("base-model/meta.json").read_text())
chars = json.loads(meta["vocab_rtl"])["chars"]
extra = sorted({c for row in processed for c in row["visual"]} - set(chars))
model = IndraRTL.from_onnx("base-model/rec_rtl.onnx")
model.add_characters(extra)
chars += extra
dataset = train.Lines(processed,chars)
x,widths,labels,label_lengths,_ = train.collate([dataset[i] for i in range(len(dataset))])
model.train()
logits,lengths = model(x,widths)
loss = F.ctc_loss(logits.log_softmax(-1).transpose(0,1),labels,lengths,label_lengths,zero_infinity=False)
assert torch.isfinite(loss)
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5)
optimizer.step()
model.eval()
model.write_onnx("base-model/rec_rtl.onnx","base-model/smoke-updated.onnx")
parity = verify_parity("base-model/smoke-updated.onnx",model)
report = {"samples":len(dataset),"finite_ctc_loss":float(loss.detach()),"all_gradients_finite":True,"updated_onnx_parity":parity,"extra_characters":extra,"max_width":int(widths.max())}
Path("local-verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False))
