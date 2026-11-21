"""Differentiable reconstruction of Indra's shipped rec_rtl.onnx graph.

Depthwise convolution BatchNorms are already fused in the ONNX initializers.
The stem BatchNorm statistics stay frozen. Export replaces float initializers in
the original graph, retaining its decoding, dynamic lengths and output contract.
"""

from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch import nn
from torch.nn import functional as F


class SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Conv1d(256, 64, 1)
        self.fc2 = nn.Conv1d(64, 256, 1)

    def forward(self, x, mask):
        pooled = (x * mask).sum(2, keepdim=True) / mask.sum(2, keepdim=True).clamp_min(1)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv1d(128, 128, 9, padding=4, groups=128)
        self.expand = nn.Conv1d(128, 256, 1)
        self.se = SE()
        self.project = nn.Conv1d(256, 128, 1)

    def forward(self, x, mask):
        y = F.hardswish(self.expand(self.dw(x)))
        return (x + self.project(self.se(y, mask))) * mask


class IndraRTL(nn.Module):
    def __init__(self, classes=1356):
        super().__init__()
        self.stem = nn.Conv2d(1, 128, (32, 4), stride=(32, 4))
        self.stem_bn = nn.BatchNorm1d(128, eps=1e-5)
        self.blocks = nn.ModuleList([Block() for _ in range(10)])
        self.head = nn.Conv1d(128, classes, 1)
        self.onnx_names = {}

    def train(self, mode=True):
        super().train(mode)
        self.stem_bn.eval()
        return self

    def forward(self, x, widths):
        x = self.stem(x[:, :, :32, :]).squeeze(2)
        lengths = ((widths + 3) // 4).clamp(max=x.shape[2])
        mask = (torch.arange(x.shape[2], device=x.device)[None, :] < lengths[:, None])[
            :, None, :
        ].to(x.dtype)
        x = F.hardswish(self.stem_bn(x)) * mask
        for block in self.blocks:
            x = block(x, mask)
        return self.head(x).transpose(1, 2), lengths

    @classmethod
    def from_onnx(cls, path):
        graph = onnx.load(str(path))
        weights = {p.name: numpy_helper.to_array(p).copy() for p in graph.graph.initializer}
        model = cls(classes=weights["rec.head.bias"].shape[0])
        mapping = {
            name: "rec." + name
            for name in model.state_dict()
            if name != "stem_bn.num_batches_tracked"
        }
        dw_nodes = [n for n in graph.graph.node if n.op_type == "Conv" and "/dw/" in n.output[0]]
        assert len(dw_nodes) == 10
        for i, node in enumerate(dw_nodes):
            mapping[f"blocks.{i}.dw.weight"] = node.input[1]
            mapping[f"blocks.{i}.dw.bias"] = node.input[2]
        state = model.state_dict()
        for name, onnx_name in mapping.items():
            state[name] = torch.from_numpy(weights[onnx_name])
        model.load_state_dict(state, strict=True)
        model.onnx_names = mapping
        model.eval()
        return model

    def write_onnx(self, original, destination):
        graph = onnx.load(str(original))
        replace = {
            onnx_name: self.state_dict()[name].detach().cpu().numpy().astype(np.float32)
            for name, onnx_name in self.onnx_names.items()
        }
        for index, initializer in enumerate(graph.graph.initializer):
            if initializer.name in replace:
                graph.graph.initializer[index].CopyFrom(
                    numpy_helper.from_array(replace[initializer.name], initializer.name)
                )
        onnx.checker.check_model(graph)
        onnx.save(graph, str(destination))

    def add_zwnj(self, space_id=1):
        """Keep all existing classes and initialize Persian half-space from space."""
        old = self.head
        head = nn.Conv1d(128, old.out_channels + 1, 1).to(old.weight.device)
        with torch.no_grad():
            head.weight[:-1].copy_(old.weight)
            head.bias[:-1].copy_(old.bias)
            head.weight[-1].copy_(old.weight[space_id])
            head.bias[-1].copy_(old.bias[space_id] - 3)
        self.head = head

    def add_characters(self, characters, space_id=1):
        old = self.head
        head = nn.Conv1d(128, old.out_channels + len(characters), 1).to(old.weight.device)
        with torch.no_grad():
            head.weight[: old.out_channels].copy_(old.weight)
            head.bias[: old.out_channels].copy_(old.bias)
            for i, char in enumerate(characters, start=old.out_channels):
                if char == "\u200c":
                    head.weight[i].copy_(old.weight[space_id])
                    head.bias[i].copy_(old.bias[space_id] - 3)
                else:
                    head.weight[i].copy_(old.weight.mean(0))
                    head.bias[i].copy_(old.bias.mean() - 3)
        self.head = head


def verify_parity(path, model=None, atol=2e-4):
    """Compare logits, IDs and lengths on unequal padded widths, before training."""
    import onnxruntime as ort

    graph = onnx.load(str(path))
    classes = next(i.dims[0] for i in graph.graph.initializer if i.name == "rec.head.bias")
    graph.graph.output.append(
        onnx.helper.make_tensor_value_info(
            "/rec/Transpose_output_0", onnx.TensorProto.FLOAT, [None, None, classes]
        )
    )
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(graph.SerializeToString(), so, providers=["CPUExecutionProvider"])
    model = model or IndraRTL.from_onnx(path)
    model.eval()
    rng = np.random.default_rng(42)
    differences = []
    for width, actual in [(128, [128, 61, 16]), (640, [637, 320, 111])]:
        x = rng.random((3, 1, 32, width), dtype=np.float32)
        widths = np.asarray(actual, dtype=np.int64)
        for i, w in enumerate(widths):
            x[i, :, :, w:] = 0
        expected_ids, expected_lengths, expected_logits = sess.run(
            ["ids", "lengths", "/rec/Transpose_output_0"], {"x": x, "widths": widths}
        )
        with torch.no_grad():
            logits, lengths = model(torch.from_numpy(x), torch.from_numpy(widths))
        result = logits.cpu().numpy()
        delta = float(np.abs(result - expected_logits).max())
        np.testing.assert_allclose(result, expected_logits, atol=atol, rtol=2e-5)
        np.testing.assert_array_equal(result.argmax(2), expected_ids)
        np.testing.assert_array_equal(lengths.cpu().numpy(), expected_lengths)
        differences.append(delta)
    return {
        "max_logit_difference": max(differences),
        "ids_identical": True,
        "lengths_identical": True,
    }


if __name__ == "__main__":
    import json

    torch.set_num_threads(2)
    base = Path(__file__).resolve().parent / "base-model/rec_rtl.onnx"
    print(json.dumps(verify_parity(base), indent=2))
    model = IndraRTL.from_onnx(base)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
